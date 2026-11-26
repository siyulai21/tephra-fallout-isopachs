"""Sensitivity of each accepted deposit to pooling a station's unresolved variation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.models.nested.reach_integrals import nodes
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.multiscale.fit import fit
from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField


def assess(job):
    record, fraction, release, output, work = job
    data, meta = audited_data(record["path"])
    latitude = meta["vent_lonlat"][1]
    if fraction == 0.5:
        import json

        if work:
            root = work / record["slug"] / "full"
            field = DirectionalField.load(root / "field.npz")
            receipt = json.loads((root / "fit.json").read_text())
        else:
            root = release / record["slug"] / "multiscale"
            field = DirectionalField.load(root / "posterior.npz")
            receipt = json.loads((root / "report.json").read_text())["fit"]
    else:
        attempts = []
        for power in (8, 10, 11):
            field, receipt = fit(data, latitude, shared_fraction=fraction, power=power)
            attempts.append(dict(power=power, **receipt))
            if receipt["passed"]:
                break
        from src.empirical_isopach.models.directional.multiscale.validation import write_json
        folder = output.parent / "resolution-attempts"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / f"{record['slug']}-{fraction:g}.json", attempts)
    exact = ~data.censored
    errors = abs(field.central(data.xy[exact]) * np.log(10) - data.log_value[exact])
    volumes = []
    for seed in (51071, 68389):
        xy, weight, *_ = nodes(13, seed, latitude)
        volumes.append(float(1e-5 * np.mean(weight * 10 ** field.central(xy))))
    return dict(
        slug=record["slug"],
        input_sha256=meta["input_sha256"],
        shared_fraction=fraction,
        measured_absolute_log_error=float(errors.mean()),
        map_km3_first=volumes[0],
        map_km3_second=volumes[1],
        passed=receipt["passed"],
        max_cdf_error=receipt["max_pooled_cdf_error"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release", type=Path)
    source.add_argument("--work", type=Path, help="Use the completed full-fit packets before plotting")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    jobs = [(r, v, args.release, args.output, args.work) for r in datasets() for v in (0.0, 0.5, 1.0)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(assess, jobs))
    pd.DataFrame(rows).to_csv(args.output, index=False)
