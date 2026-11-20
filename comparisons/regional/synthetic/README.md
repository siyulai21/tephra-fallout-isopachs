# Matched regional known-deposit trials

[Six-eruption regional comparison](../README.md)

10 fixed seeds per scenario; each model is tuned from training observations, then tested at 400 independent sites in the same hull. Conditional volume intervals are also checked against the known integral.

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

[All synthetic results and design](synthetic.json) · [Per-run comparison](comparison.csv)
