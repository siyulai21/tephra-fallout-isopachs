import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from .data import read_observations
from .model import Candidate, SpatialModel
from .validation import select_model, nested_validate
from .report import domain_bounds, grid_cells, support_flags, integrate, expected_volume, save_map


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit reproducible isopachs from real observations; ignore invented pseudo-points.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--nested-cv", action="store_true", help="Also estimate performance with outer held-out spatial regions.")
    parser.add_argument("--buffer-km", type=float, default=0.)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--volume-grid", type=int, default=24)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    args = parser.parse_args(argv)
    if not 8 <= args.grid <= 300:
        parser.error("--grid must be between 8 and 300")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to preserve previous runs.")
    data, meta = read_observations(args.csv)
    inverse = meta.pop("transform_inv")
    model, validation = select_model(data, n_folds=args.folds, buffer_km=args.buffer_km)
    bounds = np.asarray(args.bounds) if args.bounds is not None else domain_bounds(data.xy)
    xy, area = grid_cells(bounds, args.grid)
    mean, var = model.predict_log(xy)
    inside, distance = support_flags(data, xy)
    valid_alternatives = [r for r in validation["candidates"] if r["tail"] != model.candidate.tail and r["selection_score"] is not None]
    alternate = None
    alternate_mean = None
    if valid_alternatives:
        alt = min(valid_alternatives, key=lambda r: r["selection_score"])
        alternate = SpatialModel(Candidate(alt["tail"], alt["length_factor"], alt["noise_ratio"])).fit(data)
        alternate_mean, _ = alternate.predict_log(xy)
    volume = integrate(model, bounds, args.volume_grid, args.draws, args.seed)
    center, half = (bounds[:2]+bounds[2:])/2, (bounds[2:]-bounds[:2])/2
    sensitivity = []
    for factor in (1., 1.5, 2.):
        b = np.r_[center-factor*half, center+factor*half]
        sensitivity.append({"domain_factor": factor, "bounds_km": b.tolist(),
                            "selected_mean_km3": expected_volume(model, b),
                            "alternative_mean_km3": expected_volume(alternate, b) if alternate else None})
    fine_volume = expected_volume(model, bounds, 2*args.volume_grid)
    nested = nested_validate(data, outer_folds=args.folds, buffer_km=args.buffer_km) if args.nested_cv else None
    warnings = ["Intervals condition on estimated trend/covariance and model selection; they omit parameter and model uncertainty.",
                "Finite-domain volume is not total erupted volume. Unsampled tails are assumption-dependent.",
                "Legacy positive values are accepted as measurements unless flagged Pseudo; historical trace/absence encodings need provenance review."]
    if model.trend_boundary:
        warnings.append("Selected trend touches its decay constraint; extrapolated decay is weakly identified.")
    if volume["fraction_expected_volume_outside_data_hull"] > .5:
        warnings.append("More than half of expected volume lies outside the observed convex hull.")
    if nested and nested["coverage95_observations"] < .9:
        warnings.append("Nested held-out 95% observation interval coverage is below 90%; uncertainty is poorly calibrated.")
    grid_change = abs(fine_volume-volume["mean_km3"])/max(fine_volume, 1e-12)
    if grid_change > .01:
        warnings.append("Doubling the integration grid changes expected volume by more than 1%; refine --volume-grid.")
    if sensitivity[-1]["selected_mean_km3"] > 1.2*sensitivity[0]["selected_mean_km3"]:
        warnings.append("Doubling domain widths raises expected volume by more than 20%; distal volume is unresolved.")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_root = Path(__file__).parents[1]
    source_hashes = {str(p.relative_to(source_root)): sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.rglob("*.py"))}
    report = {"method_version": "0.2.0", "input": meta, "model": model.describe(),
              "alternative_model": alternate.describe() if alternate else None,
              "validation": validation, "nested_validation": nested, "volume": volume,
              "domain_sensitivity": sensitivity, "volume_grid_check": {"refined_mean_km3": fine_volume,
                  "relative_change": grid_change},
              "warnings": warnings, "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "reproducibility": {"git_base_commit": commit, "source_sha256": source_hashes,
                                  "python": platform.python_version(), "platform": platform.platform(),
                                  "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "matplotlib", "pyproj")}}}
    # Compute everything before creating artifacts, so invalid science input leaves no partial output.
    args.output.mkdir(parents=True, exist_ok=True)
    lon, lat = inverse.transform(xy[:, 0], xy[:, 1])
    z = 1.959963984540054
    table = pd.DataFrame({"x_km": xy[:, 0], "y_km": xy[:, 1], "Lon": lon, "Lat": lat,
                          "median_cm": np.exp(mean), "conditional_mean_cm": np.exp(mean+var/2),
                          "conditional_lower95_cm": np.exp(mean-z*np.sqrt(var)),
                          "conditional_upper95_cm": np.exp(mean+z*np.sqrt(var)),
                          "log_sd": np.sqrt(var), "inside_data_hull": inside,
                          "nearest_observation_km": distance, "cell_area_km2": area})
    table.to_csv(args.output/"predictions.csv", index=False)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save_map(args.output/"isopachs.png", model, xy, (args.grid, args.grid), mean, var, inside, alternate_mean)
    # Export a bounded JOINT distribution as the rigorous replacement for fake data.
    virtual_xy, _ = grid_cells(bounds, min(args.volume_grid, 24))
    vm, vc = model.predict_log(virtual_xy, covariance=True)
    np.savez_compressed(args.output/"virtual_predictions.npz", xy_km=virtual_xy, mean_log=vm, covariance_log=vc,
                        role="posterior prediction; never independent training data")
    print(json.dumps({"output": str(args.output), "model": asdict(model.candidate),
                      "n_real": len(data.xy), "ignored_pseudo": len(meta["ignored_pseudo_rows"]),
                      "conditional_domain_volume_km3": volume["conditional_quantiles_km3"],
                      "nested_validation": None if nested is None else {k: nested[k] for k in
                          ("rmse_log", "coverage95_observations", "mean_log_score")}, "warnings": warnings}, indent=2))


if __name__ == "__main__":
    main()
