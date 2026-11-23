# Mazama model results on 236 observations

The two regional fits and the global nested-isopach fit use 57 measured beds, 152 lower bounds and 27 upper bounds. These results include the six observations recovered through the dating and field-log audit. The nested result integrates the finite-spread continuation. The geographic scores use independent held-out regions. [Input and original evidence](../../../data/audits/Mazama/README.md) · [Full five-family calculation](https://github.com/siyulai21/tephra-fallout-isopachs/actions/runs/45255144401)

| Method | Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) | Held-out score ↓ | 95% coverage of withheld measured beds |
|---|---|---:|---:|---:|---:|
| GP ensemble | 1,476,452 km² observation hull | 471.43 | 367.77–1,006.00 | 1.145 | 68.4% |
| Cubic spline | 1,476,452 km² observation hull | 1,130.37 | 1,665.21–6,420,775.27 | 1.066 | 78.9% |
| Bayesian nested isopachs | Earth, conditional continuation | 262.71 | 181.80–823.50 | 1.064 | 70.2% |

The spline volume and its draw interval failed their numerical refinement checks. Integrating a pointwise median surface and taking quantiles of whole-deposit draws give different statistics. Held-out coverage is below 95% for all three methods. The support screen admits no observed regional contour band. [Regional fit and folds](regional-report.json) · [Nested uncertainty and validation](nested-extent.json) · [Independent integration checks](nested-map.json)

![Regional isopachs](regional-isopachs.png)

![Conditional nested-isopach deposit](nested-deposit.png)

[Yellowstone core detail](../../../data/audits/Mazama/YELLOWSTONE.md) · [Current evidence map](../../../data/audits/Mazama/evidence-map.png) · [Young field sections](../../../data/audits/Mazama/YOUNG_AND_FIELD_SECTIONS.md) · [Preceding 230-observation results](https://github.com/siyulai21/tephra-fallout-isopachs/tree/5fa6cae5a4e17636dd878b2776b9f71a837601e4/eruptions/mazama/audit-run)
