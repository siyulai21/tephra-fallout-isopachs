"""Publish all fixed known-truth global trials, including numerical failures."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import numpy as np

from benchmarks.mass_mixture import SCENARIOS, summarize
from src.empirical_isopach.models.mass_mixture.study import assess_posterior


LABELS = {"smooth": "Smooth exponential", "multi_lobe": "Secondary fan",
    "missing_sector": "Missing sector", "missing_core": "Invisible proximal cap",
    "matched_two_fans": "Matched spherical fans", "diffuse_global_tail": "Diffuse global tail"}


def publish(source, output):
    source, output = Path(source), Path(output)
    report = summarize(source)
    expected = {(scenario, seed) for scenario in SCENARIOS for seed in (10, 11)}
    identities = [(r["scenario"], r["seed"]) for r in report["runs"]]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Require all twelve fixed trials, without duplicates or omitted failures.")
    rows = []
    for run in sorted(report["runs"], key=lambda r: (SCENARIOS.index(r["scenario"]), r["seed"])):
        expected_fits = {"baseline", "diffuse_tail"} if run["scenario"] == "diffuse_global_tail" else {"baseline"}
        if set(run["fits"]) != expected_fits:
            raise ValueError("Missing declared stress-test assumption fit.")
        for key, fit in run["fits"].items():
            directory = source/"posteriors"/f"{run['scenario']}-{run['seed']}"/key
            for attempt in fit["attempts"]:
                path = directory/f"attempt-{attempt['attempt']}.npz"
                if sha256(path.read_bytes()).hexdigest() != attempt["posterior_sha256"]:
                    raise ValueError(f"Stress posterior checksum mismatch: {path}")
                with np.load(path) as archive:
                    samples = {"contamination": archive["contamination"]}
                attempt["posterior_assessment"] = assess_posterior(attempt["diagnostics"], samples)
            fit["diagnostics"] = fit["attempts"][-1]["posterior_assessment"]
            volume, diagnostics = fit["volume"], fit["diagnostics"]
            rows.append({"scenario": run["scenario"], "seed": run["seed"], "assumption": key,
                "truth_km3": run["truth_km3"], **{k: volume[k] for k in ("lower95", "median", "upper95")},
                "covers_truth": fit["covers_truth"], "sampling_passed": diagnostics["passed"],
                "max_rhat": diagnostics["max_label_invariant_rhat"], "divergences": diagnostics["divergences"],
                "median_relative_error": fit["median_relative_error"]})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"stress-comparison.csv", index=False)
    base = [r for r in rows if r["assumption"] == "baseline"]
    converged = [r for r in base if r["sampling_passed"]]
    report["summary"] = {"baseline_trials": len(base), "baseline_intervals_containing_truth": sum(r["covers_truth"] for r in base),
        "baseline_sampling_passed": len(converged), "converged_intervals_containing_truth": sum(r["covers_truth"] for r in converged),
        "interpretation": "Descriptive development counts only, not a calibration claim. Numerically provisional trials remain in the total and are identified separately.",
        "reporting_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"stress-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    table = "\n".join(f"| {LABELS[r['scenario']]} · {r['seed']} · {'baseline' if r['assumption'] == 'baseline' else 'diffuse-tail prior'} | {r['truth_km3']:.3g} | {r['median']:.3g} | {r['lower95']:.3g}–{r['upper95']:.3g} | {'yes' if r['covers_truth'] else '**no**'} | {'pass' if r['sampling_passed'] else '**provisional**'} |" for r in rows)
    s = report["summary"]
    (output/"stress-tests.md").write_text(f"""# Global-volume known-truth stress tests

[All-eruption results](README.md) · [Method and assumptions](../../docs/methods/probabilistic-global.md) · [Earlier development failures](development/README.md)

The final physical model and priors were fixed before these runs. There are **12 baseline trials**, six scenarios with fresh seeds 10 and 11. The two diffuse-tail cases also receive the predeclared diffuse-tail sensitivity. No trial is omitted because its interval misses the truth or its sampler fails a numerical check.

**{s['baseline_intervals_containing_truth']}/12 baseline intervals contain the known total. {s['baseline_sampling_passed']}/12 baseline fits pass numerical checks.** Among those passing, {s['converged_intervals_containing_truth']}/{s['baseline_sampling_passed']} intervals contain the truth. These small development counts do not establish 95% calibration. A numerical pass also does not establish model correctness; provisional trial intervals remain uncertain computational summaries.

| Scenario · seed · assumption | True km³ | Median km³ | Conditional 95% interval km³ | Contains truth | Sampling |
|---|---:|---:|---:|---|---|
{table}

![Estimated versus true global volume](stress.png)

## What is being challenged

- **Smooth exponential:** a simple radial deposit, sampled along twelve directions and six radii. This shape exposed the initial Gaussian/power and diffuse-weight problems; the final trials use new noise/location seeds.
- **Secondary fan:** adds a displaced anisotropic lobe. The whole field is fitted jointly, without identifying fan membership from the synthetic truth.
- **Missing sector:** removes observations from a quadrant containing a secondary fan. Unobserved mass depends on modelling assumptions.
- **Invisible proximal cap:** adds a compact positive deposit that is exactly zero at every sampled location. The data cannot identify its volume. A broad interval that includes this particular cap is not evidence that all invisible caps would be covered.
- **Matched spherical fans:** two normalized spherical components, fixed total 120 km³. It tests the model in a relatively favorable setting but is still not a draw from the full fitting prior.
- **Diffuse global tail:** a spherical power tail with exponent 0.7 and total 120 km³. Its infinite-plane analogue is nonintegrable. The Earth-normalized truth is finite; the baseline exponent restriction and the alternative allowing exponents below one are both retained.

The original four truths use an equal-area chart; the exponential/Gaussian mass omitted beyond the physical Earth chart is below 10⁻⁸⁰ of the total. The compact proximal cap is wholly inside it. The two spherical truths use exact whole-Earth normalizers. Each scenario adds independent log-observation noise with SD 0.15. True volumes are calculated analytically, not from the fitting model's posterior or a plotting window.

Large upper tails and biased medians can coexist. A conditional credible interval is not a data-only bound, and a finite prior cannot guarantee coverage of every unseen deposit. The results support continuing to report explicit assumptions and sensitivity, rather than presenting the new global model as universally calibrated or as a demonstrated improvement over GP/spline for regional prediction.

## Reproduce and inspect

```sh
python -m benchmarks.mass_mixture --output results/global-stress
python -m benchmarks.global_stress_report --source results/global-stress --output results/global-stress-report
```

The reporting command requires all twelve fixed trials and every declared alternative. [Complete priors, attempts, diagnostics and source identities](stress-report.json) · [Numerical table](stress-comparison.csv). The regional benchmark was also rerun: all six eruption fits/holdouts and all 40 regional synthetic trial records reproduce exactly; see [verification](verification.json).
""")
    figures(output, rows)
    return report


def figures(output, rows, scenario_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src.probabilistic_isopach.figures import STYLE
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 8.5), layout="constrained")
        for i, row in enumerate(rows):
            color = "#157F88" if row["covers_truth"] else "#B25335"
            ax.plot([row["lower95"]/row["truth_km3"], row["upper95"]/row["truth_km3"]], [i, i], color=color, lw=2)
            ax.scatter(row["median"]/row["truth_km3"], i, color=color, s=40,
                       marker="o" if row["sampling_passed"] else "x")
        names = LABELS if scenario_labels is None else scenario_labels
        labels = [f"{names[r['scenario']]} · {r['seed']}"+(" · diffuse prior" if r["assumption"] != "baseline" else "") for r in rows]
        ax.set(yticks=range(len(rows)), yticklabels=labels, xscale="log",
               xlabel="Estimated volume / known true volume · log scale",
               title="Whole-Earth volume stress tests\nPoints: posterior medians · lines: conditional 95% intervals")
        ax.axvline(1, color="#293D43", ls="--", lw=1.2)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
        ax.legend(handles=[Line2D([], [], color="#157F88", lw=2, label="Interval contains truth"),
            Line2D([], [], color="#B25335", lw=2, label="Interval misses truth"),
            Line2D([], [], color="#293D43", marker="x", lw=0, label="Numerically provisional")],
            loc="lower right", fontsize=9)
        fig.savefig(output/"stress.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.output)
