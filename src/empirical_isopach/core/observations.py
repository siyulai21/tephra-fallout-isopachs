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
    lower_censored: np.ndarray | None = None  # censored value is a lower, rather than upper, bound

    interval_log_upper: np.ndarray | None = None  # NaN for one-sided bounds and exact observations

    reported_minimum: np.ndarray | None = None  # reported lower bounds fitted at their thresholds

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=float)
        self.log_value = np.asarray(self.log_value, dtype=float)
        self.censored = np.asarray(self.censored, dtype=bool)
        self.log_sd = np.asarray(self.log_sd, dtype=float)
        self.row_ids = np.asarray(self.row_ids)
        n = len(self.xy)
        self.lower_censored = (np.zeros(n, dtype=bool) if self.lower_censored is None
                               else np.asarray(self.lower_censored, dtype=bool))
        self.interval_log_upper = (np.full(n, np.nan) if self.interval_log_upper is None
                                   else np.asarray(self.interval_log_upper, dtype=float))
        self.reported_minimum = (np.zeros(n, dtype=bool) if self.reported_minimum is None
                                 else np.asarray(self.reported_minimum, dtype=bool))
        if self.xy.shape != (n, 2) or any(a.shape != (n,) for a in
                (self.log_value, self.censored, self.log_sd, self.row_ids, self.lower_censored, self.interval_log_upper, self.reported_minimum)):
            raise ValueError("Observation arrays must have matching lengths and xy shape (n, 2).")
        if not all(np.all(np.isfinite(a)) for a in (self.xy, self.log_value, self.log_sd)):
            raise ValueError("Coordinates, log thicknesses, and errors must be finite.")
        if np.any(self.reported_minimum & self.censored):
            raise ValueError("A minimum fitted as a point cannot also have a censored likelihood.")
        if np.any(self.log_sd < 0):
            raise ValueError("LogSD cannot be negative.")
        if np.any(self.lower_censored & ~self.censored):
            raise ValueError("A lower bound must also be marked censored.")

        if (np.any(np.isinf(self.interval_log_upper)) or
                np.any(self.interval & (~self.lower_censored | (self.interval_log_upper <= self.log_value)))):
            raise ValueError("Intervals require two ordered finite log endpoints and a lower-bound flag.")

    @property
    def interval(self):
        return np.isfinite(self.interval_log_upper)

    def reported_bounds(self):
        """Restore geological evidence for contour support and observation symbols."""
        if not np.any(self.reported_minimum):
            return self
        return replace(self, censored=self.censored | self.reported_minimum,
                       lower_censored=self.lower_censored | self.reported_minimum, reported_minimum=None)

    def subset(self, indices):
        return replace(self, **{k: getattr(self, k)[indices] for k in
                               ("xy", "log_value", "censored", "log_sd", "row_ids", "lower_censored", "interval_log_upper", "reported_minimum")})

    def canonical(self):
        return self.subset(np.lexsort((self.interval_log_upper, self.lower_censored, self.log_sd, self.log_value, self.censored,
                                      self.xy[:, 1], self.xy[:, 0])))


IntervalObservations = Observations  # Historical public name.


def read_observations(path, *, allow_intervals=True):
    """Ignore legacy pseudo rows, including their coordinates, for all fitting.

    A non-detection requires Kind=upper_limit and Limit_cm>0. An incomplete
    deposit can provide Kind=lower_limit and a positive documented minimum.
    A thickness interval requires both positive ordered endpoints in cm.
    Missing locations are never non-detections. Vent metadata may occur on any row.
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
    allowed_kinds = {"measurement", "upper_limit", "lower_limit"}
    if allow_intervals:
        allowed_kinds.add("interval")
    elif kind.eq("interval").any():
        raise ValueError("Two-sided intervals were explicitly disabled for this reader.")
    if not set(kind).issubset(allowed_kinds):
        raise ValueError("Unknown observation kind. Use measurement, upper_limit, lower_limit or interval.")
    censored = kind.ne("measurement").to_numpy()
    interval = kind.eq("interval").to_numpy()
    for endpoint in ("Lower_cm", "Upper_cm"):
        if endpoint in real:
            supplied = ~interval & real[endpoint].notna().to_numpy()
            # Published field-survey CSVs repeat a point value in both endpoint columns.
            same_point = kind.eq("measurement") & (
                pd.to_numeric(real[endpoint], errors="coerce") == pd.to_numeric(real.Thickness, errors="coerce"))
            if np.any(supplied & ~same_point):
                raise ValueError("Unequal model endpoints require Kind=interval.")
    for other_value in ("Thickness", "Limit_cm"):
        if other_value in real and real.loc[interval, other_value].notna().any():
            raise ValueError("An interval must have blank Thickness and Limit_cm.")
    lower_censored = kind.isin(["lower_limit","interval"]).to_numpy()
    value = pd.to_numeric(real.Thickness, errors="coerce").to_numpy(dtype=float).copy()
    one_sided = censored & ~interval
    if np.any(one_sided):
        if "Limit_cm" not in real:
            raise ValueError("Censored observations require a documented positive Limit_cm.")
        value[one_sided] = real.loc[one_sided, "Limit_cm"].to_numpy(dtype=float)
    if interval.any():
        if not {"Lower_cm","Upper_cm"}.issubset(real):
            raise ValueError("A two-sided interval requires Lower_cm and Upper_cm")
        value[interval] = real.loc[interval,"Lower_cm"].to_numpy(dtype=float)
        upper = real.loc[interval,"Upper_cm"].to_numpy(dtype=float)
        if not np.all(np.isfinite(upper) & (upper>value[interval])):
            raise ValueError("Interval upper bounds must be finite and exceed the lower bound")
    if not np.all(np.isfinite(value) & (value > 0)):
        raise ValueError("Thicknesses must be positive. Encode non-detections as upper_limit with Limit_cm.")
    log_sd = real.get("LogSD", pd.Series(0., index=real.index)).fillna(0).to_numpy(dtype=float)
    fwd, inv, crs = make_laea_transform(*vent)
    xy = np.column_stack(fwd.transform(coords[:, 0], coords[:, 1]))
    args = (xy, np.log(value), censored, log_sd, real.index.to_numpy() + 2, lower_censored)
    bounds = np.full(len(real), np.nan)
    if interval.any():
        bounds[interval] = np.log(upper)
    minimum = np.zeros(len(real), dtype=bool)
    if "Reported_lower_limit_cm" in real:
        supplied = real["Reported_lower_limit_cm"]
        minimum = supplied.notna().to_numpy()
        if (np.any(minimum & censored) or
                not np.array_equal(supplied[minimum].to_numpy(dtype=float), value[minimum])):
            raise ValueError("A fitted reported minimum must equal its positive point value.")
    data = Observations(*args, interval_log_upper=bounds, reported_minimum=minimum).canonical()
    metadata = {"source": str(path), "input_sha256": sha256(path.read_bytes()).hexdigest(),
                "vent_lonlat": vent.tolist(), "projection": crs.to_string(),
                "n_measurements": int((~censored).sum()), "n_upper_limits": int((censored & ~lower_censored).sum()),
                "n_lower_limits": int((lower_censored & ~interval).sum()),
                "ignored_pseudo_rows": (df.index[is_pseudo] + 2).tolist(),
                "ignored_extra_columns": sorted(set(df) - required - {"Pseudo", "Kind", "Limit_cm", "LogSD"}),
                "transform_inv": inv}
    if minimum.any():
        metadata["fitting_policy"] = "lower-only-at-reported-minimum-v1"
        metadata["n_reported_measurements"] = int((~censored & ~minimum).sum())
        metadata["n_minima_fitted_as_measurements"] = int(minimum.sum())
        metadata["minimum_as_measurement_rows"] = (real.index[minimum] + 2).tolist()
        metadata["ignored_extra_columns"] = sorted(set(metadata["ignored_extra_columns"]) - {"Reported_lower_limit_cm"})
    if interval.any():
        metadata["n_intervals"] = int(interval.sum())
        metadata["ignored_extra_columns"] = sorted(set(metadata["ignored_extra_columns"])-{"Lower_cm","Upper_cm"})
    return data, metadata
