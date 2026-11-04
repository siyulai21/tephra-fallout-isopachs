# Regional comparison: GP ensemble and cubic spline

[Project home](../../README.md) · [Report catalogue](../README.md) · [Statistical protocol](../../docs/methods/regional-comparison.md)

Both methods are reported as peers. Every eruption uses its accepted data, identical nested geographic holdouts and one shared observation hull. No global volume enters this comparison.

| Eruption | Expected regional volume: GP / spline (km³) | Outer equal-region score: GP / spline ↓ | 95% observation coverage: GP / spline |
|---|---:|---:|---:|
| [Aso-4](../aso-4/comparison/README.md) | 207.2 / 254.1 | 1.659 / 2.066 | 83.9% / 79.3% |
| [B-Tm](../b-tm/comparison/README.md) | 30.5 / 40.2 | 1.331 / 1.814 | 97.6% / 90.5% |
| [Mazama](../mazama/comparison/README.md) | 141.6 / 132.6 | 1.272 / 1.971 | 95.6% / 82.2% |
| [Minoan](../minoan/comparison/README.md) | 27.2 / 24.8 | 1.451 / 1.601 | 96.7% / 96.7% |
| [Tambora](../tambora/comparison/README.md) | 97.9 / 107.7 | 1.620 / 1.548 | 96.2% / 96.2% |
| [YTT](../ytt/comparison/README.md) | 1196.1 / 1203.0 | 1.934 / 2.153 | 91.3% / 82.6% |

These are bulk fallout volumes inside each dataset’s observation hull, not total eruption volumes. Different eruptions have different footprints. Lower predictive scores are better; close differences across four folds do not establish superiority.

![Regional volume and held-out scores](comparison.png)

[Machine-readable real-data comparison](comparison.csv) · [Release verification](validation.json). Each eruption report includes full tuning tables, fold assignments, conditional joint-volume draws, numerical refinements and deletion sensitivity.

**Uncertainty limitation:** conditional spline volume intervals covered the known volume in only 38–50% of completed trials across these scenarios. They are not calibrated 95% confidence intervals. GP coverage also fails in some scenarios. Use the disclosed coverage and refitting sensitivity; a narrow interval is not evidence of accuracy.

## Known-truth development trials

Ten fixed seeds per scenario; each model is tuned from training observations, then tested at 400 independent sites in the same hull. Conditional volume intervals are also checked against the known integral.

| Scenario / method | Completed / attempted | Predictive score ↓ | Observation coverage | Mean absolute volume error | Volume interval coverage |
|---|---:|---:|---:|---:|---:|
| censored / Trend-regularized cubic spline | 8 / 10 | 0.441 | 90.3% | 18.2% | 38% |
| censored / GP ensemble | 8 / 10 | 0.346 | 91.8% | 17.7% | 50% |
| missing_sector / Trend-regularized cubic spline | 10 / 10 | 0.605 | 88.4% | 10.7% | 40% |
| missing_sector / GP ensemble | 10 / 10 | 0.286 | 93.2% | 10.9% | 60% |
| secondary_lobe / Trend-regularized cubic spline | 10 / 10 | 0.752 | 97.7% | 16.1% | 40% |
| secondary_lobe / GP ensemble | 10 / 10 | 0.714 | 98.6% | 13.4% | 100% |
| smooth / Trend-regularized cubic spline | 10 / 10 | 0.210 | 91.2% | 6.2% | 50% |
| smooth / GP ensemble | 10 / 10 | 0.171 | 91.7% | 6.6% | 90% |

Metrics describe completed trials only; unestimable censored trials remain in the attempted count and JSON with their failure reason. No censoring threshold or training requirement was relaxed to make these trials pass.

These small development experiments cannot certify coverage for real eruptions. Poor conditional coverage is retained rather than repaired after seeing test outcomes.

[All synthetic results and design](synthetic/synthetic.json) · [Per-run comparison](synthetic/comparison.csv)

## Reproduce

```sh
python -m benchmarks.regional --output results/regional-methods
```

No manually placed pseudo-points, smoothing settings, knot spacings or preferred method are requested. Fixed algorithmic assumptions and search limits remain explicit in the method documentation.
