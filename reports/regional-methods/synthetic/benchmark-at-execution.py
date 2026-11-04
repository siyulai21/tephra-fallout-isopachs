"""Matched regional GP-ensemble / tuned cubic-spline evaluation.

python -m benchmarks.regional --output results/regional-methods
The same frozen fitting protocol is used for real and simulated deposits.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.probabilistic_isopach.catalog import datasets
from src.probabilistic_isopach.data import Observations
from src.probabilistic_isopach.distributions import predictive_distribution
from src.probabilistic_isopach.validation import interval_metrics
from src.empirical_isopach.region import SurveyRegion, integrate_region, regional_draws
from src.empirical_isopach.spline import survey_bounds
from src.empirical_isopach.validation import METHODS, LABELS, automatic_buffer, fit_method
from src.empirical_isopach.reporting import compare_dataset, provenance, save_selection


def synthetic_case(scenario, seed):
    rng = np.random.default_rng(seed+200)
    xy = rng.uniform(-250, 250, (120, 2))
    if scenario == "missing_sector":
        xy = xy[~((xy[:, 0] > 0) & (xy[:, 1] > 0))]
    xy = xy[:60]
    def truth(points):
        values = 5-.018*np.linalg.norm(points, axis=1)+.005*points[:, 0]
        if scenario in ("secondary_lobe", "missing_sector"):
            values += 2.5*np.exp(-((points[:, 0]-130)**2+(points[:, 1]-100)**2)/(2*45**2))
        return values
    observed = truth(xy)+rng.normal(0, .25, len(xy))
    upper = observed < 1. if scenario == "censored" else np.zeros(len(xy), bool)
    lower = observed > 3. if scenario == "censored" else np.zeros(len(xy), bool)
    observed[upper], observed[lower] = 1., 3.
    data = Observations(xy, observed, upper | lower, np.zeros(len(xy)), np.arange(len(xy)), lower).canonical()
    region = SurveyRegion(xy)
    points = []
    while sum(len(p) for p in points) < 400:
        proposals = rng.uniform(-250, 250, (1000, 2))
        points.append(proposals[region.contains(proposals)])
    test = np.vstack(points)[:400]
    actual = truth(test)
    measured = actual+rng.normal(0, .25, len(test))
    heldout = Observations(test, measured, np.zeros(len(test), bool), np.zeros(len(test)), np.arange(len(test)))
    return data, region, truth, heldout, actual


def synthetic_benchmark(output, repeats=10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    def save():
        (output/"synthetic.json").write_text(json.dumps({"design": {"repeats": repeats, "noise_sd_log": .25,
            "test_locations": "400 independent uniform locations within the training-site hull",
            "censoring": "Training observations below log thickness 1 are upper limits at 1; above 3 are lower bounds at 3. Test observations remain exact.",
            "scope": "Fixed development scenarios, not samples from the fitted spline/GP and not external validation. Unestimable trials are retained explicitly.",
            "reproducibility": provenance(), "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest()},
            "runs": rows}, indent=2, allow_nan=False)+"\n")
    for scenario in ("smooth", "secondary_lobe", "missing_sector", "censored"):
        for seed in range(repeats):
            data, region, truth, heldout, actual = synthetic_case(scenario, seed)
            bounds = survey_bounds(data.xy)
            try:
                buffer, rule = automatic_buffer(data)
            except ValueError as exc:
                rows.append({"scenario": scenario, "seed": seed, "n_training": len(data.xy),
                    "n_exact": int((~data.censored).sum()), "n_upper": int((data.censored & ~data.lower_censored).sum()),
                    "n_lower": int(data.lower_censored.sum()), "status": "insufficient_training_support",
                    "methods": {m: {"status": "insufficient_training_support", "error": str(exc)} for m in METHODS}})
                save()
                print(f"Synthetic {scenario} {seed+1}/{repeats}: insufficient exact training support (both methods)", flush=True)
                continue
            class Oracle:
                def predict_log(self, points):
                    return truth(points), np.zeros(len(points))
            true_volume = integrate_region(Oracle(), region, orders=(16, 32, 64, 96, 128, 192), tolerance=1e-5)
            if not true_volume["converged"]:
                raise RuntimeError("Synthetic truth integral did not converge.")
            row = {"scenario": scenario, "seed": seed, "n_training": len(data.xy),
                   "n_exact": int((~data.censored).sum()), "n_upper": int((data.censored & ~data.lower_censored).sum()),
                   "n_lower": int(data.lower_censored.sum()), "n_test": len(heldout.xy),
                   "buffer_km": buffer, "buffer_rule": rule, "region": region.describe(),
                   "true_regional_volume_km3": true_volume["expected_km3"], "methods": {}}
            for method in METHODS:
                model, selection = fit_method(method, data, bounds, 4, buffer)
                observations = predictive_distribution(model, heldout.xy, observation=True, log_sd=heldout.log_sd)
                latent = predictive_distribution(model, heldout.xy)
                metrics = interval_metrics(heldout, observations.quantile(.5), observations.quantile(.025),
                                           observations.quantile(.975), observations.log_score(heldout.log_value))
                volume = integrate_region(model, region)
                draws = regional_draws(model, region, volume["expected_km3"], draws=2000)
                quantiles = draws["conditional_quantiles_km3"]
                result = {"status": "complete", "observations": metrics, "model": model.describe(),
                          "latent_rmse_log": float(np.sqrt(np.mean((latent.quantile(.5)-actual)**2))),
                          "latent_coverage95": float(np.mean((actual >= latent.quantile(.025)) & (actual <= latent.quantile(.975)))),
                          "regional_volume": volume, "joint_volume": draws,
                          "expected_volume_relative_error": volume["expected_km3"]/true_volume["expected_km3"]-1,
                          "median_surface_volume_relative_error": volume["median_surface_km3"]/true_volume["expected_km3"]-1,
                          "conditional_volume_interval_covers_truth": quantiles["0.025"] <= true_volume["expected_km3"] <= quantiles["0.975"],
                          "selection": selection}
                save_selection(output, f"selection/{scenario}-{seed}-{method}.csv", selection)
                row["methods"][method] = result
            rows.append(row)
            save()
            print(f"Synthetic {scenario} {seed+1}/{repeats} complete", flush=True)
    return rows


def summarize(output):
    output = Path(output)
    rows = []
    for path in sorted(output.glob("*/report.json")):
        report = json.loads(path.read_text())
        for method in METHODS:
            result = report["methods"][method]
            rows.append({"dataset": report["dataset"], "slug": path.parent.name, "method": method,
                "n_exact": report["input"]["n_measurements"], "n_lower": report["input"]["n_lower_limits"],
                "area_km2": report["region"]["area_km2"], "buffer_km": report["validation"]["buffer_km"],
                "expected_regional_km3": result["regional_volume"]["expected_km3"],
                "median_surface_km3": result["regional_volume"]["median_surface_km3"],
                "conditional_volume_lower95_km3": result["joint_volume"]["conditional_quantiles_km3"]["0.025"],
                "conditional_volume_upper95_km3": result["joint_volume"]["conditional_quantiles_km3"]["0.975"],
                "volume_converged": result["regional_volume"]["converged"],
                **{k: v for k, v in report["validation"]["methods"][method]["metrics"].items() if k != "fold_log_scores"}})
    table = pd.DataFrame(rows)
    table.to_csv(output/"comparison.csv", index=False)
    lines = ["# Regional comparison: GP ensemble and cubic spline", "",
        "Both methods are reported as peers. Every eruption uses its accepted data, identical nested geographic holdouts and one shared observation hull. No global volume enters this comparison.", "",
        "| Eruption | Expected regional volume: GP / spline (km³) | Outer equal-region score: GP / spline ↓ | 95% observation coverage: GP / spline |",
        "|---|---:|---:|---:|"]
    for slug in table.slug.unique():
        group = table[table.slug == slug].set_index("method")
        gp, spline = group.loc["stacked"], group.loc["spline"]
        lines.append(f"| [{gp['dataset']}]({slug}/README.md) | {gp.expected_regional_km3:.1f} / {spline.expected_regional_km3:.1f} | "
                     f"{gp.equal_region_log_score:.3f} / {spline.equal_region_log_score:.3f} | {gp.coverage95_observations:.1%} / {spline.coverage95_observations:.1%} |")
    lines += ["", "These are bulk fallout volumes inside each dataset’s observation hull, not total eruption volumes. Different eruptions have different footprints. Lower predictive scores are better; close differences across four folds do not establish superiority.", "",
              "![Regional volume and held-out scores](comparison.png)", "",
              "[Machine-readable real-data comparison](comparison.csv). Each eruption report includes full tuning tables, fold assignments, conditional joint-volume draws, numerical refinements and deletion sensitivity."]
    if (output/"synthetic/synthetic.json").exists():
        simulations = json.loads((output/"synthetic/synthetic.json").read_text())
        entries = []
        for run in simulations["runs"]:
            for method, result in run["methods"].items():
                if result.get("status", "complete") != "complete":
                    entries.append({"scenario": run["scenario"], "seed": run["seed"], "method": method,
                                    "completed": 0, "failure": result["error"]})
                    continue
                entries.append({"scenario": run["scenario"], "seed": run["seed"], "method": method,
                    "completed": 1, "failure": "",
                    "negative_log_score": result["observations"]["mean_log_score"],
                    "observation_coverage95": result["observations"]["coverage95_observations"],
                    "latent_coverage95": result["latent_coverage95"],
                    "volume_relative_error": result["expected_volume_relative_error"],
                    "volume_absolute_relative_error": abs(result["expected_volume_relative_error"]),
                    "volume_interval_covers_truth": result["conditional_volume_interval_covers_truth"]})
        synthetic = pd.DataFrame(entries)
        synthetic.to_csv(output/"synthetic/comparison.csv", index=False)
        summary = synthetic.groupby(["scenario", "method"]).mean(numeric_only=True)
        lines += ["", "## Known-truth development trials", "",
                  "Ten fixed seeds per scenario; each model is tuned from training observations, then tested at 400 independent sites in the same hull. Conditional volume intervals are also checked against the known integral.", "",
                  "| Scenario / method | Completed / attempted | Predictive score ↓ | Observation coverage | Mean absolute volume error | Volume interval coverage |",
                  "|---|---:|---:|---:|---:|---:|"]
        for (scenario, method), row in summary.iterrows():
            group = synthetic[(synthetic.scenario == scenario) & (synthetic.method == method)]
            lines.append(f"| {scenario} / {LABELS[method]} | {int(group.completed.sum())} / {len(group)} | {row.negative_log_score:.3f} | {row.observation_coverage95:.1%} | "
                         f"{row.volume_absolute_relative_error:.1%} | {row.volume_interval_covers_truth:.0%} |")
        lines += ["", "Metrics describe completed trials only; unestimable censored trials remain in the attempted count and JSON with their failure reason. No censoring threshold or training requirement was relaxed to make these trials pass.", "",
                  "These small development experiments cannot certify coverage for real eruptions. Poor conditional coverage is retained rather than repaired after seeing test outcomes.", "",
                  "[All synthetic results and design](synthetic/synthetic.json) · [Per-run comparison](synthetic/comparison.csv)"]
    lines += ["", "## Reproduce", "", "```sh", "python -m benchmarks.regional --output results/regional-methods", "```", "",
              "No manually placed pseudo-points, smoothing settings, knot spacings or preferred method are requested. Fixed algorithmic assumptions and search limits remain explicit in the method documentation."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.probabilistic_isopach.figures import STYLE, TEAL, AMBER
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        names = table.dataset.unique()
        for method, shift, color in (("stacked", -.16, TEAL), ("spline", .16, AMBER)):
            subset = table[table.method == method].set_index("dataset").loc[names]
            axes[0].barh(np.arange(len(names))+shift, subset.expected_regional_km3, height=.3, color=color, label=LABELS[method])
            axes[1].barh(np.arange(len(names))+shift, subset.equal_region_log_score, height=.3, color=color)
        for ax in axes:
            ax.set(yticks=np.arange(len(names)), yticklabels=names)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.15)
        axes[0].set(xscale="log", xlabel="Expected regional bulk fallout (km³; log scale)", title="One shared footprint per eruption")
        axes[1].set(xlabel="Outer equal-region negative log score (lower is better)", title="Identical geographic holdouts")
        axes[0].legend(fontsize=9, loc="lower right")
        fig.suptitle("Two empirical methods · matched regional evaluation", fontsize=16)
        for suffix in ("png", "svg"):
            fig.savefig(output/f"comparison.{suffix}", dpi=180)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", help="One catalogued eruption; default runs all six")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    if args.summarize_only:
        summarize(args.output)
        return
    if args.repeats < 1:
        parser.error("Use at least one synthetic replicate")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.synthetic_only:
        chosen = [r for r in datasets() if args.dataset is None or args.dataset.lower() in (r["slug"], r["name"].lower())]
        if not chosen:
            parser.error("Unknown eruption")
        for record in chosen:
            compare_dataset(record["path"], args.output/record["slug"], name=record["name"],
                            progress=lambda s: print(s, flush=True))
    if not args.skip_synthetic:
        if (args.output/"synthetic/synthetic.json").exists():
            parser.error("Synthetic results already exist; use a fresh destination")
        synthetic_benchmark(args.output/"synthetic", args.repeats)
    if not args.synthetic_only:
        summarize(args.output)


if __name__ == "__main__":
    main()
