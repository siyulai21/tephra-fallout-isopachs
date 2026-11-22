# B-Tm · Directional ensemble

The principal bulk volume is **33.5 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **23.5–77.1 km³** across complete posterior draws.

![B-Tm, Directional ensemble. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

Fixing the fan directions gives a median-map volume of 34.6 km³. Solid fragments require thicknesses on opposite sides of the contour and at least 90% posterior crossing probability. [Whole-Earth context](azimuth/world.png) · [Earlier 0.1 cm extent and its edge checks](azimuth/map-display.json)

## Reach from the volcano

![Fixed and uncertain fan directions near the source](azimuth/median-isopachs.png)

Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.

| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |
|---|---:|---:|---:|---:|
| N | 18.5 | 17.4 | 134 | 118 |
| NE | 33.1 | 33.0 | 3,586 | 2,853 |
| E | 462.2 | 462.4 | 65,483 | 63,063 |
| SE | 30.5 | 30.1 | 4,398 | 4,198 |
| S | 23.8 | 21.5 | 215 | 175 |
| SW | 19.6 | 18.2 | 157 | 133 |
| W | 26.2 | 22.5 | 245 | 193 |
| NW | 19.6 | 18.3 | 159 | 134 |

![Probability that a deposit reaches ten centimetres](azimuth/exceedance.png)

The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.

## Withheld observations

Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. Coverage counts exact withheld thicknesses inside the uncertain-direction 95% predictive interval. A dash marks an unavailable score or coverage.

| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geographic regions | 4 | 1.273 | 1.243 | 3 | 1.253 | 1.234 | 60/64 (93.8%) |
| Source quadrants | 3 | 2.657 | 1.947 | 3 | 2.657 | 1.947 | 56/64 (87.5%) |

[Predictions for every estimable held-out site](azimuth/heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.

## Inspected archives

The principal fit includes the interpreted upper limits below. Each core enters once. Other sites provide an occurrence check. The interval describes the modeled latent thickness at each site.

| Site | Interpreted upper limit (cm) | Median thickness (cm) | Conditional 95% interval (cm) |
|---|---:|---:|---:|
| Lake Sihailongwan | 0.06 | 0.0146 | 4.71e-09–0.298 |
| Quanyang peat | 0.64 | 0.358 | 0.00123–3.25 |
| Lake Kushu | 0.45 | 0.217 | 5.81e-38–2.39 |
| Ulleung Basin 13PT-P4_026 | Occurrence only | 2.95e-09 | 1.44e-39–0.0527 |
| YK14-01 PC09 | Occurrence only | 0.753 | 0.361–1.64 |
| NT13-19 PL01 | Occurrence only | 0.618 | 0.221–1.58 |

## Checks and reproducibility

The shared contour audit found eligible bands under none of the screened support policies. Audit status is completed. [All contour decisions and deletion failures](azimuth/complete-contour-audit.json)

The spatial integration check passed at 16,384 points per scramble. The map used up to 64 posterior draws per chain and structure. Resolved contour areas passed the split-packet check. The joint-volume quantiles passed the split check.

All six full fits passed their sampling diagnostics.

[Numerical report, all sector volumes and sensitivities](azimuth/report.json) · [Complete fields and per-field volumes](azimuth/fields.npz) · [Principal map grids](azimuth/median-grids.npz) · [Near-source comparison grids](azimuth/map-grid.npz) · [Fragment decisions](azimuth/fragment-decisions.csv) · [Bracketed coordinates](azimuth/bracketed-fragments.csv) · [File hashes](azimuth/artifacts.json)

## Related directional studies

- [Earlier fixed-direction ensemble and its separate calibration](ensemble/README.md)
- [Single-structure and proximal-cap sensitivities](single-structure/README.md)
- [Measured-fan continuation and inspected-site evidence](bearing/README.md)
