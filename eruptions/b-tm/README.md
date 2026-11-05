# B-Tm: data, global and regional results

[All eruptions](../README.md) · [Active methods](../../methods/README.md) · [Add or revise data](../../docs/NEW_ERUPTION.md)

## Global fallout

The joint global posterior has median **344 km³** and a **79.5–17,485 km³ conditional 95% credible interval** for bulk fallout. All accepted fans are fitted together. This whole-Earth estimate uses no closing pseudo-data or arbitrary calculation region. [Read the global report, assumptions and sensitivity refits](results/global/README.md). The wide upper tail reflects deposits that remain poorly constrained by observations.

**Local support audit:** no complete isopach band passes the support and deletion checks. The standard local footprint covers 52,431 km². Read the [local partial volumes and rejected-contour checks](results/support/README.md) before interpreting the broader hull results below. The separate [conditional global analysis](results/global/README.md) now estimates the whole deposit under explicit assumptions; this contour-screen decision is unchanged.

## Regional reconstruction

**GP ensemble and automatic cubic spline are reported as peers.** Both use the same accepted data, geographic holdouts and observation hull.

[Read the full comparison](results/README.md) · [Compare all six eruptions](../../comparisons/regional/README.md) · [Method definitions](../../docs/METHODS.md)

84 exact measurements and 0 lower bounds; regional area 963,645 km².

| Quantity | GP ensemble | Cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 30.46 | 40.16 |
| Outer equal-region negative log score ↓ | 1.331 | 1.814 |

![Paired regional isopachs](results/isopachs.png)

These volumes cover only the observation hull. It can include unsampled gaps and exclude proximal deposits; it is not total eruption volume or dense-rock equivalent. Conditional intervals omit fitted-parameter and selection uncertainty. Read withheld coverage and geographic-deletion sensitivity in the full report.

## Evidence and calculation

| Follow the calculation | File |
|---|---|
| Accepted input | [Thickness observations](../../data/eruptions/B-Tm_tephra.csv) |
| Source decisions | [Audit or supplied-flag status](../../data/eruptions/README.md) |
| Whole-deposit mass and uncertainty | [Global posterior report](results/global/README.md) |
| Local maps and partial volumes | [Regional report](results/README.md) |
| Fits, validation and provenance | [Complete numerical record](results/report.json) |
| Row-by-row outer predictions | [Held-out predictions](results/heldout_predictions.csv) |
| Automatic tuning | [Candidate tables](results/selection/README.md) |

The 2026-11-04 regional rerun reproduced accepted inputs, fitted settings, held-out predictions and matched-hull volumes exactly. The new global companion is published separately. [Rerun verification](../../comparisons/global/verification.json). [Previous matched-hull report](../../archive/2026-11-03-regional/b-tm/README.md).

## Earlier calculations

[GP-only global ensemble](../../archive/reports/b-tm/global/stacked/README.md) · [Earlier selected GP](../../archive/reports/b-tm/global/selected/README.md) · [Rectangular regional history](../../archive/reports/b-tm/regional/README.md)

Historical numerical artifacts retain their original method, source hashes and integration domain. They are not used to rank the current regional methods.
