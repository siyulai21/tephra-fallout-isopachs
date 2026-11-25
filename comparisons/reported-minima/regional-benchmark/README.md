# Matched regional trials with lower limits fitted as points

In the censored scenario, reports above log thickness 3 are fitted as point values at 3. Reports below 1 contribute upper-censored probabilities at 1. The test sites retain their simulated measured thickness.

The design uses ten fixed seeds per scenario and each model is tuned from training observations, then tested at 400 independent sites in the same hull. Conditional volume intervals are also checked against the known integral.

| Scenario / method | Completed / attempted | Predictive score ↓ | Observation coverage | Mean absolute volume error | Volume interval coverage |
|---|---:|---:|---:|---:|---:|
| censored / Trend-regularized cubic spline | 10 / 10 | 1.130 | 82.8% | 33.7% | 0% |
| censored / GP ensemble | 10 / 10 | 0.852 | 85.0% | 34.2% | 0% |
| missing_sector / Trend-regularized cubic spline | 10 / 10 | 0.605 | 88.4% | 10.7% | 40% |
| missing_sector / GP ensemble | 10 / 10 | 0.286 | 93.2% | 10.9% | 60% |
| secondary_lobe / Trend-regularized cubic spline | 10 / 10 | 0.752 | 97.7% | 16.1% | 40% |
| secondary_lobe / GP ensemble | 10 / 10 | 0.714 | 98.6% | 13.4% | 100% |
| smooth / Trend-regularized cubic spline | 10 / 10 | 0.210 | 91.2% | 6.2% | 50% |
| smooth / GP ensemble | 10 / 10 | 0.171 | 91.7% | 6.6% | 90% |

Both methods completed all 40 trials. The predictive scores and coverage use independent withheld locations within the simulation hull. Conditional volume-interval coverage varies across these fixed scenarios. [Per-run scores](comparison.csv) · [Design, full results and numerical checks](synthetic.json)
