# YTT: data and regional results

[All eruptions](../README.md) · [Active methods](../../methods/README.md) · [Add or revise data](../../docs/NEW_ERUPTION.md)

**New support audit:** no complete isopach band passes the support and deletion checks. The standard local footprint covers 1,196,058 km². Read the [local partial volumes and rejected-contour checks](results/support/README.md) before interpreting the broader hull results below. No preferred global volume is reported.

**GP ensemble and automatic cubic spline are reported as peers.** Both use the same accepted data, geographic holdouts and observation hull.

[Read the full comparison](results/README.md) · [Compare all six eruptions](../../comparisons/regional/README.md) · [Method definitions](../../docs/METHODS.md)

46 exact measurements and 0 lower bounds; regional area 17,719,965 km².

| Quantity | GP ensemble | Cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 1196.14 | 1203.01 |
| Outer equal-region negative log score ↓ | 1.934 | 2.153 |

![Paired regional isopachs](results/isopachs.png)

These volumes cover only the observation hull. It can include unsampled gaps and exclude proximal deposits; it is not total eruption volume or dense-rock equivalent. Conditional intervals omit fitted-parameter and selection uncertainty. Read withheld coverage and geographic-deletion sensitivity in the full report.

The [audited principal selection](../../data/audits/YTT/README.md) still contains 46 observations. No source decisions or input values changed.

## Evidence and calculation

| Follow the calculation | File |
|---|---|
| Accepted input | [Thickness observations](../../data/eruptions/YTT_tephra.csv) |
| Source decisions | [Audit or supplied-flag status](../../data/audits/YTT/README.md) |
| Current maps and interpretation | [Regional report](results/README.md) |
| Fits, validation and provenance | [Complete numerical record](results/report.json) |
| Row-by-row outer predictions | [Held-out predictions](results/heldout_predictions.csv) |
| Automatic tuning | [Candidate tables](results/selection/README.md) |

Both methods were rerun on 2026-11-04. Accepted inputs, fitted settings, held-out predictions and matched-hull volumes reproduce the preceding comparison; the new support audit adds local evidence and extrapolation checks. [Previous matched-hull report](../../archive/2026-11-03-regional/ytt/README.md).

## Earlier calculations

[GP-only global ensemble](../../archive/reports/ytt/global/stacked/README.md) · [Earlier selected GP](../../archive/reports/ytt/global/selected/README.md) · [Rectangular regional history](../../archive/reports/ytt/regional/README.md)

Historical numerical artifacts retain their original method, source hashes and integration domain. They are not used to rank the current regional methods.
