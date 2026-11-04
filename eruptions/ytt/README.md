# YTT: data and regional results

[All eruptions](../README.md) · [Active methods](../../methods/README.md) · [Add or revise data](../../docs/NEW_ERUPTION.md)

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

The current published result was generated before the repository reorganization. Its numerical artifacts and recorded execution hashes are preserved unchanged; moving a file does not create a new analysis.

## Earlier calculations

[GP-only global ensemble](../../archive/reports/ytt/global/stacked/README.md) · [Earlier selected GP](../../archive/reports/ytt/global/selected/README.md) · [Rectangular regional history](../../archive/reports/ytt/regional/README.md)

Historical numerical artifacts retain their original method, source hashes and integration domain. They are not used to rank the current regional methods.
