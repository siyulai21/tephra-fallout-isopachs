"""Known-truth development tests of support screening and contour eligibility.

python -m benchmarks.support --output results/support-trials
All model tuning is repeated in training folds; thresholds are never retuned
from these outcomes. These deliberately small stress tests are not calibration.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.core.geometry import survey_bounds
from src.empirical_isopach.models.registry import active_methods, method_metadata
from src.empirical_isopach.validation import nested_comparison, fit_method
from src.empirical_isopach.region import SurveyRegion, integrate_region
from src.empirical_isopach.support import SupportMesh
from src.empirical_isopach.contour_support import audit_contours, median_grid, level_polygons, shape_change
from src.empirical_isopach.reporting import provenance, save_selection


def synthetic_case(scenario, seed):
    rng = np.random.default_rng(47100+seed)
    radii, theta = np.meshgrid(np.arange(30., 181., 30.), np.arange(12)*2*np.pi/12)
    xy = np.column_stack((radii.ravel()*np.cos(theta.ravel()), radii.ravel()*np.sin(theta.ravel())))
    xy += rng.uniform(-3, 3, xy.shape)
    if scenario == "missing_sector":
        xy = xy[~((xy[:, 0] > 0) & (xy[:, 1] > 0))]
    if scenario == "missing_core":
        xy = xy[np.linalg.norm(xy, axis=1) > 100]

    def truth(points):
        r = np.linalg.norm(points, axis=1)
        t = np.exp(5-r/60)
        if scenario in ("multi_lobe", "missing_sector"):
            t += 300*np.exp(-.5*((points[:, 0]-110)/25)**2-.5*((points[:, 1]-110)/40)**2)
        if scenario == "missing_core":
            # This positive C2 cap is identically zero at every sampled site.
            t += 500*np.maximum(1-(r/75)**2, 0)**3
        return np.log(t)

    values = truth(xy)+rng.normal(0, .15, len(xy))
    data = Observations(xy, values, np.zeros(len(xy), bool), np.zeros(len(xy)), np.arange(len(xy))).canonical()
    total = 2*np.pi*np.exp(5)*60**2*1e-5
    if scenario in ("multi_lobe", "missing_sector"):
        total += 2*np.pi*300*25*40*1e-5
    if scenario == "missing_core":
        total += 500*np.pi*75**2/4*1e-5
    return data, truth, total


def run(output, repeats=3):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output folder.")
    output.mkdir(parents=True)
    report = {"design": {"repeats": repeats, "seeds": list(range(repeats)),
        "sampling": "Six radii (30..180 km), twelve directions, independent +/-3 km coordinate jitter; missing-sector/core exclusions precede observation generation.",
        "noise_sd_log": .15, "model_metadata": method_metadata(), "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": provenance(), "scope": "Fixed development stress tests, not independent calibration. Geometry and all thresholds frozen before inspecting results. Missing-core truth adds a cap invisible to every observation. No preferred global estimate is expected."}, "runs": []}
    for scenario in ("smooth", "multi_lobe", "missing_sector", "missing_core"):
        for seed in range(repeats):
            data, truth, total = synthetic_case(scenario, seed)
            region, bounds = SurveyRegion(data.xy), survey_bounds(data.xy)
            deletion = {m: [] for m in active_methods()}
            validation = nested_comparison(data, region, bounds,
                model_callback=lambda method, fold, model: deletion[method].append(model))
            mesh = SupportMesh(data)
            class Oracle:
                def predict_log(self, xy):
                    return truth(xy), np.zeros(len(xy))
            actual_volume = integrate_region(Oracle(), mesh, orders=(8, 16, 24, 36, 48, 64, 96), tolerance=1e-5)
            if not actual_volume["converged"]:
                raise RuntimeError("Known truth footprint integral did not converge.")
            row = {"scenario": scenario, "seed": seed, "n": len(data.xy),
                "support": mesh.describe(), "true_supported_km3": actual_volume["expected_km3"],
                "true_total_infinite_plane_km3": float(total), "validation": validation, "methods": {}}
            for method in active_methods():
                model, selection = fit_method(method, data, bounds, 4, validation["buffer_km"])
                volume = integrate_region(model, mesh)
                audit, grid, polygons, _ = audit_contours(model, data, bounds, deletion[method], {"standard": mesh})
                _, truth_polygons = level_polygons(median_grid(Oracle(), bounds, 513), [r["level_cm"] for r in audit["levels"]])
                comparisons = []
                for entry in audit["levels"]:
                    t = entry["level_cm"]
                    comparisons.append({"level_cm": t, "eligible": entry["policies"]["standard"]["eligible"],
                        "true_contour_closed_in_window": t in truth_polygons,
                        "shape_error": shape_change(polygons.get(t), truth_polygons.get(t))})
                prefix = f"selection/{scenario}-{seed}-{method}"
                save_selection(output, prefix+"-all.csv", selection)
                for split in validation["methods"][method]["selections"]:
                    save_selection(output, prefix+f"-outer-{split['fold']}.csv", split["selection"])
                row["methods"][method] = {"model": model.describe(), "selection": selection,
                    "support_volume": volume, "expected_volume_relative_error": volume["expected_km3"]/actual_volume["expected_km3"]-1,
                    "median_volume_relative_error": volume["median_surface_km3"]/actual_volume["expected_km3"]-1,
                    "contours": audit, "truth_comparisons": comparisons}
            report["runs"].append(row)
            (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(f"Support stress test {scenario} {seed+1}/{repeats} complete", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use at least one repeat")
    run(args.output, args.repeats)


if __name__ == "__main__":
    main()
