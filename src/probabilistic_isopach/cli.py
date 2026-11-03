import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from .data import read_observations
from .model import Candidate, SpatialModel
from .validation import select_model, nested_validate
from .report import domain_bounds, grid_cells, support_flags, integrate, expected_volume, save_map
from .distributions import predictive_distribution


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "atlas":
        from .atlas import main as atlas_main
        return atlas_main(argv[1:])
    if argv and argv[0] == "serve":
        from .workbench import main as workbench_main
        return workbench_main(argv[1:])
    if argv and argv[0] == "fit":
        argv = argv[1:]
    parser = argparse.ArgumentParser(description="Fit reproducible isopachs from real observations; ignore invented pseudo-points.",
        epilog="Other workflows: tephra-isopachs serve | tephra-isopachs atlas --output results/atlas")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--method", choices=("selected", "stacked"), default="selected",
                        help="selected: established single-model workflow; stacked: experimental spatial ensemble with observation calibration.")
    parser.add_argument("--nested-cv", action="store_true", help="Also estimate performance with outer held-out spatial regions.")
    parser.add_argument("--buffer-km", type=float, default=0.)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--volume-grid", type=int, default=48)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--name", help="Display name; defaults to the CSV stem.")
    parser.add_argument("--map-grid", type=int, default=481, help="Odd resolution for closed-contour maps (65–1201).")
    parser.add_argument("--levels", type=float, nargs="+", help="Positive increasing isopach thicknesses in cm.")
    parser.add_argument("--max-extent-km", type=float, default=6000., help="Maximum absolute easting/northing of the contour map.")
    parser.add_argument("--no-publication", action="store_true", help="Only write compact numerical outputs and the legacy diagnostic map.")
    parser.add_argument("--atlas-parent", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 8 <= args.grid <= 300:
        parser.error("--grid must be between 8 and 300")
    if not 8 <= args.volume_grid <= 60 or not 20 <= args.draws <= 100000:
        parser.error("--volume-grid must be 8–60; --draws must be 20–100000")
    if args.seed < 0 or args.folds < 2:
        parser.error("Use a nonnegative seed and at least two folds")
    if not 65 <= args.map_grid <= 1201 or args.map_grid % 2 != 1:
        parser.error("--map-grid must be an odd number from 65 to 1201")
    if not 100 <= args.max_extent_km <= 8000:
        parser.error("--max-extent-km must be between 100 and 8000")
    if args.levels is not None and (not np.isfinite(args.levels).all() or min(args.levels) <= 0 or np.any(np.diff(args.levels) <= 0)):
        parser.error("--levels must be finite, positive and strictly increasing")
    from .catalog import resolve_dataset
    args.csv = resolve_dataset(args.csv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to preserve previous runs.")
    data, meta = read_observations(args.csv)
    repository_root = Path(__file__).resolve().parents[2]
    def portable_path(value):
        path = Path(value).resolve()
        return str(path.relative_to(repository_root)) if path.is_relative_to(repository_root) else path.name
    meta["source"] = portable_path(args.csv)
    inverse = meta.pop("transform_inv")
    model, validation = select_model(data, n_folds=args.folds, buffer_km=args.buffer_km, method=args.method)
    bounds = np.asarray(args.bounds) if args.bounds is not None else domain_bounds(data.xy)
    xy, area = grid_cells(bounds, args.grid)
    distribution = predictive_distribution(model, xy)
    mean = distribution.quantile(.5)
    moment_mean, var = distribution.moments()
    inside, distance = support_flags(data, xy)
    reference_tail = getattr(model, "dominant_tail", model.candidate.tail)
    valid_alternatives = [r for r in validation["candidates"] if r["tail"] != reference_tail and r["selection_score"] is not None]
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
    nested = nested_validate(data, outer_folds=args.folds, buffer_km=args.buffer_km, method=args.method) if args.nested_cv else None
    uncertainty = (model.describe()["uncertainty"] if args.method == "stacked" else
                   "Intervals condition on estimated trend/covariance and model selection; they omit parameter and model uncertainty.")
    warnings = [uncertainty,
                "Finite-domain volume is not total erupted volume. Unsampled tails are assumption-dependent.",
                "Legacy positive values are accepted as measurements unless flagged Pseudo; historical trace/absence encodings need provenance review."]
    if args.method == "stacked":
        warnings.append("Experimental ensemble: weights and new-observation discrepancy are fitted from spatial holdouts; only outer validation assesses their performance. No coverage guarantee. Discrepancy does not widen latent maps or volume intervals.")
        if validation["stacking"]["calibration"]["at_search_boundary"]:
            warnings.append("Observation-discrepancy variance reached its search boundary; calibration remains unresolved.")
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
        commit = subprocess.check_output(["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_root = Path(__file__).parents[1]
    source_hashes = {str(p.relative_to(source_root)): sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_root.rglob("*.py"))}
    report = {"method_version": "0.4.0" if args.method == "stacked" else "0.2.0", "input": meta, "model": model.describe(),
              "alternative_model": alternate.describe() if alternate else None,
              "validation": validation, "nested_validation": nested, "volume": volume,
              "domain_sensitivity": sensitivity, "volume_grid_check": {"refined_mean_km3": fine_volume,
                  "relative_change": grid_change},
              "warnings": warnings, "settings": {k: portable_path(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "reproducibility": {"git_base_commit": commit, "source_sha256": source_hashes,
                                  "asset_sha256": {str(p.relative_to(source_root)): sha256(p.read_bytes()).hexdigest()
                                                   for p in sorted((Path(__file__).parent/"assets").glob("*.geojson"))},
                                  "python": platform.python_version(), "platform": platform.platform(),
                                  "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "matplotlib", "pyproj")}}}
    # Compute everything before creating artifacts, so invalid science input leaves no partial output.
    args.output.mkdir(parents=True, exist_ok=True)
    lon, lat = inverse.transform(xy[:, 0], xy[:, 1])
    observations = predictive_distribution(model, xy, observation=True)
    adjusted = predictive_distribution(model, xy, observation=True, calibrated=True)
    table = pd.DataFrame({"x_km": xy[:, 0], "y_km": xy[:, 1], "Lon": lon, "Lat": lat,
                          "median_cm": np.exp(mean), "conditional_mean_cm": np.exp(distribution.log_expected_thickness()),
                          "conditional_lower95_cm": np.exp(distribution.quantile(.025)),
                          "conditional_upper95_cm": np.exp(distribution.quantile(.975)),
                          "new_observation_lower95_cm": np.exp(observations.quantile(.025)),
                          "new_observation_upper95_cm": np.exp(observations.quantile(.975)),
                          "diagnostic_adjusted_observation_lower95_cm": np.exp(adjusted.quantile(.025)),
                          "diagnostic_adjusted_observation_upper95_cm": np.exp(adjusted.quantile(.975)),
                          "between_model_variance_log": distribution.weights@((distribution.means-moment_mean)**2),
                          "log_sd": np.sqrt(var), "inside_data_hull": inside,
                          "nearest_observation_km": distance, "cell_area_km2": area})
    table.to_csv(args.output/"predictions.csv", index=False)
    if args.no_publication:
        save_map(args.output/"isopachs.png", model, xy, (args.grid, args.grid), mean, var, inside, alternate_mean)
    else:
        from .publication import publish_dataset
        print(f"Preparing closed contours and checking volume integrals for {args.name or args.csv.stem}…", flush=True)
        publish_dataset(args.output, args.name or args.csv.stem.replace("_tephra", ""), model, alternate,
                        inverse, report, args.map_grid, args.levels, args.max_extent_km)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    # Export a bounded JOINT distribution as the rigorous replacement for fake data.
    virtual_xy, _ = grid_cells(bounds, min(args.volume_grid, 24))
    vm, vc = model.predict_log(virtual_xy, covariance=True)
    joint = {"xy_km": virtual_xy, "mean_log": vm, "covariance_log": vc,
             "role": "joint latent prediction; never independent training data"}
    if hasattr(model, "models"):
        components = [m.predict_log(virtual_xy, covariance=True) for m in model.models]
        joint.update(distribution="Gaussian mixture; mean/covariance alone are only moments",
                     component_weights=model.weights, component_mean_log=np.array([p[0] for p in components]),
                     component_covariance_log=np.array([p[1] for p in components]))
    else:
        joint.update(distribution="Gaussian conditional approximation")
    np.savez_compressed(args.output/"virtual_predictions.npz", **joint)
    print(json.dumps({"output": str(args.output), "model": asdict(model.candidate),
                      "n_real": len(data.xy), "ignored_pseudo": len(meta["ignored_pseudo_rows"]),
                      "conditional_domain_volume_km3": volume["conditional_quantiles_km3"],
                      "nested_validation": None if nested is None else {k: nested[k] for k in
                          ("rmse_log", "coverage95_observations", "mean_log_score")}, "warnings": report["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
