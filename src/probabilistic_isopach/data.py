"""Strict observation semantics and backward-compatible CSV reading."""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from src.Laea import make_laea_transform


@dataclass
class Observations:
    xy: np.ndarray
    log_value: np.ndarray  # log thickness for detections, log limit for upper limits
    censored: np.ndarray
    log_sd: np.ndarray  # additional known measurement error, not fitted nugget
    row_ids: np.ndarray

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=float)
        self.log_value = np.asarray(self.log_value, dtype=float)
        self.censored = np.asarray(self.censored, dtype=bool)
        self.log_sd = np.asarray(self.log_sd, dtype=float)
        self.row_ids = np.asarray(self.row_ids)
        n = len(self.xy)
        if self.xy.shape != (n, 2) or any(a.shape != (n,) for a in
                (self.log_value, self.censored, self.log_sd, self.row_ids)):
            raise ValueError("Observation arrays must have matching lengths and xy shape (n, 2).")
        if not all(np.all(np.isfinite(a)) for a in (self.xy, self.log_value, self.log_sd)):
            raise ValueError("Coordinates, log thicknesses, and errors must be finite.")
        if np.any(self.log_sd < 0):
            raise ValueError("LogSD cannot be negative.")

    def subset(self, indices):
        return replace(self, **{k: getattr(self, k)[indices] for k in
                               ("xy", "log_value", "censored", "log_sd", "row_ids")})

    def canonical(self):
        return self.subset(np.lexsort((self.log_sd, self.log_value, self.censored,
                                      self.xy[:, 1], self.xy[:, 0])))


def read_observations(path):
    """Ignore legacy pseudo rows, including their coordinates, for all fitting.

    A non-detection requires Kind=upper_limit and Limit_cm>0. Missing locations
    are never converted to non-detections. Vent metadata may occur on any row.
    """
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Lon", "Lat", "Thickness", "Vent Lon", "Vent Lat"}
    if not required.issubset(df):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    vent_rows = df[["Vent Lon", "Vent Lat"]].dropna(how="all")
    if vent_rows.empty or vent_rows.isna().any().any():
        raise ValueError("Supply a complete Vent Lon, Vent Lat pair in at least one row.")
    vents = vent_rows.to_numpy(dtype=float)
    if not np.all(np.isfinite(vents)) or not np.allclose(vents, vents[0], rtol=0, atol=1e-8):
        raise ValueError("Vent coordinates must be finite and consistent across rows.")
    vent = vents[0]
    if abs(vent[0]) > 180 or abs(vent[1]) > 90:
        raise ValueError("Vent longitude/latitude are outside geographic bounds.")
    pseudo = df.get("Pseudo", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    pseudo = pseudo.replace({"1.0": "1", "0.0": "0"})
    allowed = {"", "no", "false", "0", "yes", "true", "1"}
    if not set(pseudo).issubset(allowed):
        raise ValueError("Pseudo must be blank, Yes/No, True/False, or 1/0.")
    is_pseudo = pseudo.isin({"yes", "true", "1"})
    real = df.loc[~is_pseudo].copy()
    coords = real[["Lon", "Lat"]].to_numpy(dtype=float)
    if (not np.all(np.isfinite(coords)) or np.any(abs(coords[:, 0]) > 180)
            or np.any(abs(coords[:, 1]) > 90)):
        raise ValueError("Every real observation needs valid longitude and latitude.")
    kind = real.get("Kind", pd.Series("measurement", index=real.index)).fillna("measurement").str.strip().str.lower()
    if not set(kind).issubset({"measurement", "upper_limit"}):
        raise ValueError("Kind must be measurement or upper_limit; traces need documented limits.")
    censored = kind.eq("upper_limit").to_numpy()
    value = pd.to_numeric(real.Thickness, errors="coerce").to_numpy(dtype=float).copy()
    if np.any(censored):
        if "Limit_cm" not in real:
            raise ValueError("Upper limits require a documented positive Limit_cm.")
        value[censored] = real.loc[kind.eq("upper_limit"), "Limit_cm"].to_numpy(dtype=float)
    if not np.all(np.isfinite(value) & (value > 0)):
        raise ValueError("Thicknesses must be positive. Encode non-detections as upper_limit with Limit_cm.")
    log_sd = real.get("LogSD", pd.Series(0., index=real.index)).fillna(0).to_numpy(dtype=float)
    fwd, inv, crs = make_laea_transform(*vent)
    xy = np.column_stack(fwd.transform(coords[:, 0], coords[:, 1]))
    data = Observations(xy, np.log(value), censored, log_sd, real.index.to_numpy() + 2).canonical()
    metadata = {"source": str(path), "input_sha256": sha256(path.read_bytes()).hexdigest(),
                "vent_lonlat": vent.tolist(), "projection": crs.to_string(),
                "n_measurements": int((~censored).sum()), "n_upper_limits": int(censored.sum()),
                "ignored_pseudo_rows": (df.index[is_pseudo] + 2).tolist(),
                "ignored_extra_columns": sorted(set(df) - required - {"Pseudo", "Kind", "Limit_cm", "LogSD"}),
                "transform_inv": inv}
    return data, metadata
