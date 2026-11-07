"""Publish every declared source-focused development trial, including failures."""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.source_fans import SCENARIOS, summarize
from benchmarks.global_stress_report import LABELS, figures


def publish(source, output):
    source, output = Path(source), Path(output)
    report = summarize(source)
    identities = [(r["scenario"], r["seed"]) for r in report["runs"]]
    expected = {(s, seed) for s in SCENARIOS for seed in (20, 21)}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Require all twelve declared trials without duplicates.")
    rows, draws = [], []
    for run in sorted(report["runs"], key=lambda r: (SCENARIOS.index(r["scenario"]), r["seed"])):
        expected_fits = {"baseline", "diffuse_tail"} if run["scenario"] == "diffuse_global_tail" else {"baseline"}
        if set(run["fits"]) != expected_fits:
            raise ValueError("Every declared assumption fit is required.")
        for key, fit in run["fits"].items():
            directory = source/"posteriors"/f"{run['scenario']}-{run['seed']}"/key
            if "source_checks" not in fit:
                raise ValueError("Run the current source_fans driver to verify angular normalization.")
            # Check every resolution and attempt, including failed numerical runs.
            stages = [(directory/r["directory"], r["fit"]) for r in fit["normalization_refinements"]]
            stages.append(((directory/fit["posterior_file"]).parent, fit))
            for folder, stage in stages:
                for attempt in stage["attempts"]:
                    path = folder/f"attempt-{attempt['attempt']}.npz"
                    if sha256(path.read_bytes()).hexdigest() != attempt["posterior_sha256"]:
                        raise ValueError(f"Posterior checksum mismatch: {path}")
            path = directory/fit["posterior_file"]
            if sha256(path.read_bytes()).hexdigest() != fit["posterior_sha256"]:
                raise ValueError(f"Final posterior checksum mismatch: {path}")
            with np.load(path) as archive:
                volume_draws = np.exp(archive["log_volume"])
            volume, diagnostics = fit["volume"], fit["diagnostics"]
            np.testing.assert_allclose(np.quantile(volume_draws, [.025, .5, .975]),
                [volume[k] for k in ("lower95", "median", "upper95")], rtol=1e-12)
            chain, draw = np.indices(volume_draws.shape)
            draws.append(pd.DataFrame({"scenario":run["scenario"], "seed":run["seed"],
                "assumption":key, "chain":chain.ravel(), "draw":draw.ravel(),
                "volume_km3":volume_draws.ravel()}))
            rows.append({"scenario":run["scenario"], "seed":run["seed"], "assumption":key,
                "truth_km3":run["truth_km3"], **{k:volume[k] for k in ("lower95", "median", "upper95")},
                "covers_truth":fit["covers_truth"], "sampling_passed":diagnostics["passed"],
                "sampling_only_passed":diagnostics["sampling_only_passed"],
                "normalization_passed":diagnostics["normalization_passed"],
                "angular_nodes":fit["source_checks"]["normalization"]["angular_nodes_fitted"],
                "max_rhat":diagnostics["max_label_invariant_rhat"], "divergences":diagnostics["divergences"],
                "median_relative_error":fit["median_relative_error"]})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"stress-comparison.csv", index=False)
    (output/"stress-volume-draws.csv.gz").write_bytes(gzip.compress(
        pd.concat(draws).to_csv(index=False).encode(), mtime=0))
    base = [r for r in rows if r["assumption"] == "baseline"]
    passed = [r for r in base if r["sampling_passed"]]
    report["summary"] = {"baseline_trials":len(base),
        "baseline_intervals_containing_truth":sum(r["covers_truth"] for r in base),
        "baseline_numerically_passed":len(passed),
        "passing_intervals_containing_truth":sum(r["covers_truth"] for r in passed),
        "interpretation":"Development counts, not external validation or a 95% calibration claim. Failed trials remain visible.",
        "reporting_source_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"stress-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    labels = {**LABELS, "matched_two_fans":"Displaced spherical truth"}
    table = "\n".join(f"| {labels[r['scenario']]} · {r['seed']} · {r['assumption']} | {r['truth_km3']:.3g} | {r['median']:.3g} | {r['lower95']:.3g}–{r['upper95']:.3g} | {'yes' if r['covers_truth'] else '**no**'} | {'pass' if r['sampling_passed'] else '**provisional**'} |" for r in rows)
    s = report["summary"]
    (output/"stress-tests.md").write_text(f"""# Source-focused model: known-truth challenges

[Source and volume review](README.md) · [Shape assumptions](../../docs/methods/source-anchored-fans.md)

These are all **12 declared baseline trials**, six scenarios with seeds 20 and 21, plus the two declared diffuse-tail alternatives. No trial-specific prior, target volume or closing boundary is supplied. Every failure is retained. The three-fan source-focused model is fitted to each complete collection of observations without using true fan membership.

**{s['baseline_intervals_containing_truth']}/12 baseline intervals contain the true total; {s['baseline_numerically_passed']}/12 pass the sampling and mass-normalization checks.** Among numerically passing fits, {s['passing_intervals_containing_truth']}/{s['baseline_numerically_passed']} intervals contain the truth. This small development suite does not establish 95% calibration or superior global-volume accuracy. Intervals from failed samplers are provisional computational summaries.

| Scenario · seed · prior | True bulk km³ | Median km³ | Conditional 95% interval km³ | Contains truth | Numerical checks |
|---|---:|---:|---:|---|---|
{table}

![Estimated and known global volumes](stress.png)

## What the scenarios test

The smooth exponential gives the constraint a favorable case. The secondary fan and missing-sector cases add a displaced anisotropic lobe; they can violate nonincreasing thickness along source rays. The invisible proximal cap contributes positive mass while being exactly zero at every sampled site. Its amplitude cannot be inferred from those observations. The displaced spherical truth has total 120 km³ and uses the preceding freely displaced model; despite the historical scenario identifier `matched_two_fans`, it is **not matched to the new source-focused model**. The diffuse global tail has exponent 0.7 and total 120 km³; both the restricted baseline and the alternative allowing exponents below one are shown.

The original four truths use an equal-area chart and analytic total masses; the tiny mass outside the physical Earth chart is below 10⁻⁸⁰ of the total. The compact cap is wholly inside the Earth chart. The last two use whole-Earth normalized kernels. Independent log-observation noise has SD 0.15. The truth is never obtained from the fitted posterior or the map extent. Shape violations and hidden mass are deliberate challenges, not grounds to omit trials.

Requiring a source maximum fixes one unsupported geometric freedom. It does not establish a universal law, identify unseen proximal caps, recover missing sectors, or ensure correct distal thinning. A broad interval that contains one invisible cap cannot guarantee coverage of every possible cap. These limitations apply to the B-Tm interpretation even when its real-data sampler and geographic prediction checks pass.

## Reproduce and audit

```sh
python -m benchmarks.source_fans --output results/source-stress
python -m benchmarks.source_stress_report --source results/source-stress --output results/source-stress-report
```

Four independent chains, the fixed retry protocol, every attempted resolution and diagnostics for every complete run are recorded in the [full report](stress-report.json). Angular integration is independently refined for every retained posterior; inadequate resolution triggers a complete refit without changing the physical priors. [Numerical table](stress-comparison.csv) · [Complete final volume chains with chain identities](stress-volume-draws.csv.gz). Full parameter samples, including failed attempts, remain in the reproducible local run caches; the public sample file contains volume draws. No chain trimming, convergence-selected pooling or volume-based choice is used.
""")
    figures(output, rows, scenario_labels=labels)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.output)
