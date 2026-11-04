"""Sensitivity to spline fitting-domain padding at a fixed support footprint.

This is a refitting sensitivity, not an outer-test comparison or a selector.
The plotting window never enters the integral. The numerical basis rectangle
can still affect the fitted spline, and that dependence must remain visible.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.observations import read_observations
from src.empirical_isopach.support import SupportMesh
from src.empirical_isopach.region import integrate_region
from src.empirical_isopach.validation import fit_method
from src.empirical_isopach.reporting import provenance, save_selection


def run(base, output):
    base, output = Path(base), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh destination.")
    output.mkdir(parents=True)
    result = {"protocol": "support-domain-sensitivity-v1", "padding_fractions": [.025, .05, .1, .2],
        "scope": "Same observations, standard local footprint and final-fit geographic tuning folds. Spline completely retuned at each rectangle. GP has no fitting rectangle; its reference is unchanged. This does not evaluate new outer prediction performance or select a preferred rectangle.",
        "provenance": provenance(), "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "runs": []}
    for record in datasets():
        parent = base/record["slug"]/"report.json"
        baseline = json.loads(parent.read_text())
        support_path = base/record["slug"]/"support/report.json"
        support = json.loads(support_path.read_text())
        data, meta = read_observations(record["path"])
        if meta["input_sha256"] != baseline["input"]["input_sha256"]:
            raise ValueError("Input differs from matched comparison.")
        mesh = SupportMesh(data)
        lo, hi = data.xy.min(axis=0), data.xy.max(axis=0)
        runs = []
        for fraction in (.025, .05, .1, .2):
            bounds = np.r_[lo-fraction*(hi-lo), hi+fraction*(hi-lo)]
            if fraction == .05:
                volume = support["methods"]["spline"]["footprint_volumes"]["standard"]
                model = baseline["methods"]["spline"]["model"]
                selection = {"reference": "parent report; original 5% padding"}
            else:
                fitted, selection = fit_method("spline", data, bounds, 4, baseline["validation"]["buffer_km"])
                volume = integrate_region(fitted, mesh)
                model = fitted.describe()
                save_selection(output, f"selection/{record['slug']}-{fraction:g}.csv", selection)
            runs.append({"padding_fraction": fraction, "fit_bounds_km": bounds.tolist(),
                         "volume": volume, "model": model, "selection": selection})
        result["runs"].append({"dataset": record["name"], "slug": record["slug"],
            "input_sha256": meta["input_sha256"], "parent_report_sha256": sha256(parent.read_bytes()).hexdigest(),
            "parent_support_sha256": sha256(support_path.read_bytes()).hexdigest(),
            "area_km2": mesh.area, "gp_expected_km3": support["methods"]["stacked"]["footprint_volumes"]["standard"]["expected_km3"],
            "spline": runs})
        (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(f"Fitting-domain sensitivity {record['name']} complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)
