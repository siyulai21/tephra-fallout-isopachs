# Ilopango TBJ (total primary airfall)

The five reconstructions use the same 168 accepted fallout localities. Field reports give 65 measured thicknesses, 84 lower limits, 5 upper limits and 14 two-sided ranges. Each lower limit enters the fit as a point at its reported value, including the 41 beyond 25 km. Upper limits and ranges retain their respective one-sided and two-sided probabilities. [Source audit](../../data/audits/Ilopango-TBJ/field-logs/README.md) · [Reported table](../../data/eruptions/Ilopango_TBJ_total_airfall.csv) · [Fitting table](../../data/eruptions/Ilopango_TBJ_total_airfall_minimum_fit.csv)

The site table pairs each source classification with its fitting value, geographic fold, mapped thickness and held-out score. It includes the predictive median or cumulative probability at the site value where the method saves it. A fitted minimum can underestimate total fallout at an incomplete section. [Reported and fitted sites](results/minimum-fit-2026-11-25/reported-and-fitted-sites.csv) · [Changes by method and coverage by source type](results/minimum-fit-2026-11-25/minimum-fit-comparison.csv) · [Fitting convention](../../docs/methods/reported-minimum-fitting.md)

## Volumes and geographic prediction

Each estimate integrates the median thickness on its map. The conditional range integrates the pointwise 25th and 75th percentiles of thickness over the same area. The 50% probability refers to thickness at each location. A single probability for total volume does not follow from these endpoints.

| Method | Integration area | Median map (km³) | Q25–Q75 thickness maps (km³) | Held-out score ↓ | 95% predictive coverage |
|---|---|---:|---:|---:|---:|
| GP ensemble | observation hull | **17.3** | 10.2–30.1 | 1.799 | 97.3% |
| Cubic spline | observation hull | **27.6** | 19.8–38.9 | 1.757 | 98.7% |
| [Multiscale directional ensemble](results/minimum-fit-2026-11-25/directional/README.md) | whole Earth | **40.3** | 24.3–75.5 | 1.734 | 93.3% |
| [Spherical mixture](results/minimum-fit-2026-11-25/spherical-mixture/README.md) | whole Earth | **90.4** | 51.1–175 | 1.793 | 90.6% |
| [Bayesian nested footprints](results/minimum-fit-2026-11-25/nested-isopachs/README.md) | whole Earth | **47.1** | 31.7–76.9 | 1.788 | 89.9% |

Lower scores indicate better geographic prediction. Coverage counts held-out measurements and reported minima fitted as points. A dagger marks a fit, integration or validation warning.

## Full deposits and near-vent detail

Each full map follows the thickest power-of-two contour enclosing at least 95% of the fitted locations, with a 1 cm minimum. The title gives the actual count, including when that minimum prevents 95% coverage. Global volumes include the thinner tail. The innermost line is the thickest doubling level enclosing an accepted location. The near-vent window expands from the proximal sites until eligible contours are visible. The curve sums all patches at each displayed thickness.

![Ilopango TBJ (total primary airfall) · Multiscale directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/minimum-fit-2026-11-25/directional/deposit.png)

![Ilopango TBJ (total primary airfall) · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/minimum-fit-2026-11-25/spherical-mixture/deposit.png)

![Ilopango TBJ (total primary airfall) · Bayesian nested footprints. Full deposit, near-vent detail and thickness versus square root of area.](results/minimum-fit-2026-11-25/nested-isopachs/deposit.png)

## Inside the observation hull

![Ilopango TBJ (total primary airfall) · GP ensemble and cubic spline on a common scale and the full observation hull](results/minimum-fit-2026-11-25/isopachs.png)

The GP and spline integrate over the dashed hull. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/minimum-fit-2026-11-25/README.md)

## Nested geometry and field agreement

The nearly concentric proximal circles become eccentric, approximately north–south contours. Their orientation changes continuously toward the northwest–southeast outer envelope. The 256 cm summit and successive 128, 64 and 32 cm contours stay connected and smoothly spaced; the full view no longer attaches a round core abruptly to an oblique outer shape. The northwestern 1 cm continuation becomes substantially longer, however. It is a remaining extrapolation weakness, not a newly established deposit boundary.

Eastern records SM1-P38 and SM1-P30 improve, and the two-sided SM1-P63 interval receives a less extreme prediction. The thin northern SM1-P118 measurement and northeastern SM1-P116 interval worsen. The latter remains a strong conflict between the fitted field and bounded evidence. Measured log RMSE improves from 1.021 to 1.011, while median absolute log error worsens from 0.537 to 0.596 and the bounded negative log probability worsens from 1.522 to 1.572. Median interval violations fall from 11 to 10. Geographic score changes from 1.761 to 1.788. Centered-core, circular-core, altered-focus and prior-reweighting trials did not establish a general solution and remain in the archive; no case-specific cutoff was imposed.

[Three-scale comparison, site identities and contour diagnostics](../../comparisons/nested-isopachs/adaptive-growth/visual-review.md#ilopango-tbj)

## Evidence and checks

[Matched predictions by locality](results/minimum-fit-2026-11-25/geographic-site-validation.csv) · [Numerical checks and file identities](results/minimum-fit-2026-11-25/release.json) · [Spherical fits and computing revisions](results/minimum-fit-2026-11-25/spherical-mixture/workflow-provenance.json) · [Regional support](results/minimum-fit-2026-11-25/support/README.md) · [Whole-area sensitivity](results/minimum-fit-2026-11-25/nested-whole-area/README.md)

[Previous maps and geological assessment](previous-minimum-treatment.md) · [Change at all distances across eruptions](../../comparisons/reported-minima/README.md)
