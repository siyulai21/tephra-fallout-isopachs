"""Recover frozen multi-fan and distal deposits without exposing their truth during fitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.wrapped_angular.regional import Posterior
from src.empirical_isopach.models.wrapped_angular.field import RegionalField


def run(folder):
    meta = json.loads((folder / "identity.json").read_text())
    obs = folder / "observations.npz"
    if sha256(obs.read_bytes()).hexdigest() != meta["frozen_input_npz_sha256"]:
        raise ValueError("Frozen synthetic input changed")
    with np.load(obs) as f:
        data = Observations(**dict(f))
    fit = Posterior.fit(data, 0.0)
    evaluate = RegionalField.from_fit(fit).coherent(128, 86243)
    rows = []
    for scramble in (51071, 68389):
        quadrature = folder / f"quadrature-13-{scramble}.npz"
        if sha256(quadrature.read_bytes()).hexdigest() != meta["evaluation_sha256"][quadrature.name]:
            raise ValueError("Frozen evaluation field changed")
        with np.load(quadrature) as f:
            xy, weights = f["xy"], f["weights"] * 1.0e-5 / len(f["xy"])
            truth = f["truth_log_cm"]
        grid = np.concatenate([evaluate(chunk) for chunk in np.array_split(xy, 16)], axis=-1)
        volume = np.exp(grid) @ weights
        q = np.quantile(grid, [0.025, 0.5, 0.975], axis=(0, 1))
        medmap = np.exp(q[1]) @ weights
        low, median, high = np.quantile(volume, [0.025, 0.5, 0.975])
        for label, actual, true_volume in zip(
            meta["truth_labels"], truth, meta["truth_volume_km3"], strict=True
        ):
            thick = actual >= np.log(0.5)
            for level in [0.5, 2.0, 8.0, 32.0]:
                real, estimate = actual >= np.log(level), q[1] >= np.log(level)
                union = weights @ (real | estimate)
                rows.append(
                    dict(
                        family="wrapped-angular-covariance",
                        network=folder.name,
                        truth=label,
                        input_sha256=meta["frozen_input_npz_sha256"],
                        scramble=scramble,
                        level_cm=level,
                        truth_km3=true_volume,
                        draw_q025=low,
                        draw_median=median,
                        draw_q975=high,
                        volume_covered=low <= true_volume <= high,
                        median_map_km3=medmap,
                        population_relative_difference=(np.median(volume[0]) - np.median(volume[1])) / median,
                        area_iou=weights @ (real & estimate) / union if union else 1.0,
                        latent_coverage95=(weights @ (thick & (actual >= q[0]) & (actual <= q[2])))
                        / (weights @ thick)
                        if thick.any()
                        else np.nan,
                    )
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=Path("comparisons/wrapped-angular-covariance/known-inputs")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [run(path.parent) for path in sorted(args.inputs.glob("*/identity.json"))]
    pd.concat(results, ignore_index=True).to_csv(args.output, index=False)
