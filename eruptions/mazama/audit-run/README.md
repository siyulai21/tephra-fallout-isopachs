# Mazama model rerun on 230 audited observations

The regional GP, cubic spline and global nested-isopach fits below use the same registered input, SHA-256 `8ff013f2f0026d5eef7c0ca76e0d12584b4fbf11861160e972c9d0a81f847bfe`. The directional ensemble, spherical prior suite and source-sensitivity publication are in progress. The existing comparison on the case home is the preceding release.

| Fit | Domain | Integrated median surface, km³ | Conditional 95% volume interval, km³ |
|---|---|---:|---:|
| GP ensemble | 1,476,452 km² observation hull | 536.14 | 381.43–920.31 |
| Cubic spline | Same hull | 1,187.83 | 1,798.53–8,133,342.98 |
| Bayesian nested isopachs | Earth, conditional continuation | 282.14 | 184.84–766.72 |

The regional held-out log RMSE is 4.060 for the GP and 4.838 for the spline. Their 95% observation intervals cover 40.0% and 72.7% of exact observations in the geographic tests. The spline integration fails its refinement criterion; the complete-draw interval and integral of its pointwise median are different statistics. Sparse directions, archival transfer and numerical sensitivity matter for interpretation. [Regional fit and folds](regional-report.json) · [Nested interval and validation](nested-extent.json) · [Nested map integration](nested-map.json)

![Matched regional isopachs](regional-isopachs.png)

![Conditional nested-isopach deposit](nested-deposit.png)

[Source ledger](../../../data/audits/Mazama/revised-locality-ledger.csv) · [Accepted observations](../../../data/eruptions/Mazama_reviewed.csv) · [Yellowstone core detail](../../../data/audits/Mazama/YELLOWSTONE.md)
