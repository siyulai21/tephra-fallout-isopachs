# Sakurajima Satsuma (P14) · Directional ensemble

The principal bulk volume is **7.21 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **3.58–15.7 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Sakurajima Satsuma (P14), Directional ensemble. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 4 cm footprint includes 55 of 57 modeled locations. The full volume also includes fallout thinner than the displayed footprint. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## Reach from the volcano

![Fixed and uncertain fan directions near the source](azimuth/median-isopachs.png)

Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.

| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |
|---|---:|---:|---:|---:|
| N | 48.2 | 46.6 | 941 | 835 |
| NE | 38.0 | 35.6 | 602 | 563 |
| E | 52.1 | 50.9 | 1,180 | 1,168 |
| SE | 85.8 | 71.3 | 3,295 | 2,647 |
| S | 111.6 | 100.9 | 4,596 | 3,978 |
| SW | 69.7 | 64.5 | 1,765 | 1,328 |
| W | 53.4 | 53.4 | 924 | 840 |
| NW | 42.5 | 36.1 | 707 | 520 |

![Probability that a deposit reaches ten centimetres](azimuth/exceedance.png)

The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.

## Withheld observations

Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. Coverage counts exact withheld thicknesses inside the uncertain-direction 95% predictive interval. A dash marks an unavailable score or coverage.

| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geographic regions | 4 | 1.028 | 1.164 | 3 | 0.949 | 1.073 | 48/50 (96.0%) |
| Source quadrants | 4 | 1.246 | 1.406 | 4 | 1.246 | 1.406 | 50/50 (100.0%) |

[Predictions for every estimable held-out site](azimuth/heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.

## Independent archive predictions

These locations have reported ash occurrences or tentative correlations. The fitted thickness data supply the predictions. The intervals describe latent bulk thickness at each location.

| Archive | Identification | Median thickness (cm) | Conditional 95% interval (cm) |
|---|---|---:|---:|
| Tawaraishi No. 1 | secure | 0.611 | 0.0013–4.3 |
| Kuwanoki | secure | 12.7 | 7.58–19.1 |
| U1429C NK457 | secure | 1.14 | 0.467–2.8 |
| Huguangyan HGY1026 | secure | 4.84e-06 | 8.39e-28–0.0204 |
| Suigetsu SG14-1554 | tentative | 0.000176 | 7.11e-14–0.0794 |

## Checks and reproducibility

The shared contour audit found eligible bands under none of the screened support policies. Audit status is completed. [All contour decisions and deletion failures](azimuth/complete-contour-audit.json)

The spatial integration check passed at 8,192 points per scramble. The map used up to 128 posterior draws per chain and structure. Resolved contour areas passed the split-packet check. The joint-volume quantiles passed the split check.

All six full fits passed their sampling diagnostics.

[Numerical report, all sector volumes and sensitivities](azimuth/report.json) · [Complete fields and per-field volumes](azimuth/fields.npz) · [Principal map grids](azimuth/median-grids.npz) · [Near-source comparison grids](azimuth/map-grid.npz) · [Fragment decisions](azimuth/fragment-decisions.csv) · [Bracketed coordinates](azimuth/bracketed-fragments.csv) · [File hashes](azimuth/artifacts.json)

## Related directional studies

- [Earlier fixed-direction ensemble and its separate calibration](ensemble/README.md)
- [Single-structure and proximal-cap sensitivities](single-structure/README.md)
- [Measured-fan continuation and inspected-site evidence](bearing/README.md)
