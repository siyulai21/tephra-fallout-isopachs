"""Portable, versioned posterior for replaying any geographic query or whole field."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import numpy as np
from src.probabilistic_isopach.global_volume import EarthDomain
from .field import Coefficients, RegionalField, Deposit, CoherentFields

FAMILY = "wrapped-angular-covariance"
PROTOCOL = "wrapped-angular-cumulative-thinning-v1"


def write(path, deposit, meta):
    reg = deposit.regional
    arrays = dict(
        protocol=PROTOCOL,
        input_sha256=meta["input_sha256"],
        vent_lonlat=meta["vent_lonlat"],
        covariance=reg.covariance,
        weights=reg.weights,
        axis=reg.axis,
        designs=np.array([g.design for g in reg.groups]),
        source=json.dumps(deposit.source),
        prior_only=deposit.prior_only,
    )
    # Source parameters were updated conditional on these exact regional realizations.
    # Retain their coefficients across changes in eigenvector signs or BLAS implementations.
    bank = deposit.bank
    arrays["bank_count"] = bank.count
    arrays["bank_designs"] = np.array([g[2] for g in bank.groups])
    for i, (pop, index, _, shape, beta) in enumerate(bank.groups):
        arrays.update(
            {f"bank_pop_{i}": pop, f"bank_index_{i}": index, f"bank_shape_{i}": shape, f"bank_beta_{i}": beta}
        )
    if deposit.params is not None:
        arrays.update(source_params=deposit.params, regional_centre=deposit.centres)
    for i, g in enumerate(reg.groups):
        for key in ("index", "shape", "beta", "covariance"):
            arrays[f"{key}_{i}"] = getattr(g, key)
    with NamedTemporaryFile(dir=Path(path).parent, delete=False) as tmp:
        try:
            np.savez_compressed(tmp, **arrays)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise
    os.replace(tmp.name, path)


def read(path, meta):
    with np.load(path, allow_pickle=False) as f:
        if (
            str(f["protocol"]) != PROTOCOL
            or str(f["input_sha256"]) != meta["input_sha256"]
            or not np.array_equal(f["vent_lonlat"], meta["vent_lonlat"])
        ):
            raise ValueError("Posterior protocol, fitting edition or vent differs")
        groups = [
            Coefficients(tuple(row), *(f[f"{name}_{i}"] for name in ("index", "shape", "beta", "covariance")))
            for i, row in enumerate(f["designs"])
        ]
        reg = RegionalField(float(f["axis"]), f["weights"], groups, str(f["covariance"]))
        draws = [
            (
                int(f[f"bank_pop_{i}"]),
                f[f"bank_index_{i}"],
                tuple(row),
                f[f"bank_shape_{i}"],
                f[f"bank_beta_{i}"],
            )
            for i, row in enumerate(f["bank_designs"])
        ]
        return Deposit(
            reg,
            json.loads(str(f["source"])),
            CoherentFields(reg.axis, reg.covariance, int(f["bank_count"]), draws),
            f.get("source_params"),
            f.get("regional_centre"),
            bool(f["prior_only"]),
        )


class MapView:
    statistic = "Pointwise median primary fallout thickness"

    def __init__(self, deposit, latitude):
        self.deposit = deposit
        self.domain = EarthDomain.wgs84(latitude)

    def values(self, xy, probabilities):
        xy = np.asarray(xy)
        mask = np.isfinite(xy).all(axis=1) & (self.domain.rho(xy) <= self.domain.rho_max)
        arr = np.full((len(probabilities), len(xy)), np.nan)
        if mask.any():
            arr[:, mask] = self.deposit.quantiles(xy[mask], probabilities, prior_power=12) / np.log(10.0)
        return arr

    def central(self, xy):
        return self.values(xy, (0.5,))[0]
