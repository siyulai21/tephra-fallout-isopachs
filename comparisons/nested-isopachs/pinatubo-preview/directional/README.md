# Pinatubo 1991 Layer C · Directional ensemble

The principal bulk volume is **4.92 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **3.2–11 km³** across complete posterior draws. **Numerical warning on the fitted deposit distribution.**

![Pinatubo 1991 Layer C, Directional ensemble. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

Fixing the fan directions gives a median-map volume of 5.27 km³. Solid fragments require thicknesses on opposite sides of the contour and at least 90% posterior crossing probability. [Whole-Earth context](azimuth/world.png) · [Earlier 0.1 cm extent and its edge checks](azimuth/map-display.json)

## Reach from the volcano

![Fixed and uncertain fan directions near the source](azimuth/median-isopachs.png)

Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.

| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |
|---|---:|---:|---:|---:|
| N | 18.1 | 17.8 | 126 | 121 |
| NE | 18.3 | 18.3 | 121 | 121 |
| E | 22.2 | 22.2 | 180 | 180 |
| SE | 22.6 | 22.6 | 224 | 224 |
| S | 30.7 | 30.7 | 440 | 440 |
| SW | 48.1 | 44.7 | 1,018 | 983 |
| W | 71.0 | 71.1 | 1,458 | 1,407 |
| NW | 20.5 | 20.2 | 204 | 195 |

![Probability that a deposit reaches ten centimetres](azimuth/exceedance.png)

The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.

## Withheld observations

Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. Coverage counts exact withheld thicknesses inside the uncertain-direction 95% predictive interval. A dash marks an unavailable score or coverage.

| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geographic regions | 4 | 2.425 | 1.804 | 2 | 4.227 | 2.883 | 95/102 (93.1%) |
| Source quadrants | 4 | 1.674 | 1.365 | 0 | — | — | 85/102 (83.3%) |

[Predictions for every estimable held-out site](azimuth/heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.

## Checks and reproducibility

The shared contour audit found eligible bands under none of the screened support policies. Audit status is completed. [All contour decisions and deletion failures](azimuth/complete-contour-audit.json)

The spatial integration check passed at 8,192 points per scramble. The map used up to 64 posterior draws per chain and structure. Resolved contour areas passed the split-packet check. The joint-volume quantiles passed the split check.

Full-fit sampling warnings occurred for narrow_fans, reference, narrow_missing_fan, broad_missing_fan.

[Numerical report, all sector volumes and sensitivities](azimuth/report.json) · [Complete fields and per-field volumes](azimuth/fields.npz) · [Principal map grids](azimuth/median-grids.npz) · [Near-source comparison grids](azimuth/map-grid.npz) · [Fragment decisions](azimuth/fragment-decisions.csv) · [Bracketed coordinates](azimuth/bracketed-fragments.csv) · [File hashes](azimuth/artifacts.json)

## Related directional studies

- [Earlier fixed-direction ensemble and its separate calibration](ensemble/README.md)
- [Single-structure and proximal-cap sensitivities](single-structure/README.md)
- [Measured-fan continuation and inspected-site evidence](bearing/README.md)
