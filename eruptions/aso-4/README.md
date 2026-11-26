# Aso-4

The five reconstructions use the same 87 accepted fallout localities. Field reports give 83 measured thicknesses, 3 lower limits, 1 upper limits and 0 two-sided ranges. Each lower limit enters the fit as a point at its reported value, including the 3 beyond 25 km. Upper limits and ranges retain their respective one-sided and two-sided probabilities. [Source audit](../../data/audits/Aso-4/README.md) · [Reported table](../../data/eruptions/Aso-4_reviewed.csv) · [Fitting table](../../data/eruptions/Aso-4_reviewed_minimum_fit.csv)

The site table pairs each source classification with its fitting value, geographic fold, mapped thickness and held-out score. It includes the predictive median or cumulative probability at the site value where the method saves it. A fitted minimum can underestimate total fallout at an incomplete section. [Reported and fitted sites](results/minimum-fit-2026-11-25/reported-and-fitted-sites.csv) · [Changes by method and coverage by source type](results/minimum-fit-2026-11-25/minimum-fit-comparison.csv) · [Fitting convention](../../docs/methods/reported-minimum-fitting.md)

## Volumes and geographic prediction

Each estimate integrates the median thickness on its map. The conditional range integrates the pointwise 25th and 75th percentiles of thickness over the same area. The 50% probability refers to thickness at each location. A single probability for total volume does not follow from these endpoints.

| Method | Integration area | Median map (km³) | Q25–Q75 thickness maps (km³) | Held-out score ↓ | 95% predictive coverage |
|---|---|---:|---:|---:|---:|
| GP ensemble | observation hull | **171** | 119–244 | 1.531 | 93.0% |
| Cubic spline | observation hull | **198** | 144–285 | 1.694 | 90.7% |
| [Multiscale directional ensemble](results/minimum-fit-2026-11-25/directional/README.md) | whole Earth | **541** | 333–1,049 | 1.225 | 97.7% |
| [Spherical mixture](results/minimum-fit-2026-11-25/spherical-mixture/README.md) | whole Earth | **449** | 267–916 | 1.600 | 94.2% |
| [Bayesian nested footprints](results/minimum-fit-2026-11-25/nested-isopachs/README.md) | whole Earth | **2,068** | 834–5,646 | 1.364 † | 93.0% |

Lower scores indicate better geographic prediction. Coverage counts held-out measurements and reported minima fitted as points. A dagger marks a fit, integration or validation warning.

## Full deposits and near-vent detail

Each full map follows the thickest power-of-two contour enclosing at least 95% of the fitted locations, with a 1 cm minimum. The title gives the actual count, including when that minimum prevents 95% coverage. Global volumes include the thinner tail. The innermost line is the thickest doubling level enclosing an accepted location. The near-vent window expands from the proximal sites until eligible contours are visible. The curve sums all patches at each displayed thickness.

![Aso-4 · Multiscale directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/minimum-fit-2026-11-25/directional/deposit.png)

![Aso-4 · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/minimum-fit-2026-11-25/spherical-mixture/deposit.png)

![Aso-4 · Bayesian nested footprints. Full deposit, near-vent detail and thickness versus square root of area.](results/minimum-fit-2026-11-25/nested-isopachs/deposit.png)

## Inside the observation hull

![Aso-4 · GP ensemble and cubic spline on a common scale and the full observation hull](results/minimum-fit-2026-11-25/isopachs.png)

The GP and spline integrate over the dashed hull. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/minimum-fit-2026-11-25/README.md)

## Evidence and checks

[Matched predictions by locality](results/minimum-fit-2026-11-25/geographic-site-validation.csv) · [Spherical fits and computing revisions](results/minimum-fit-2026-11-25/spherical-mixture/workflow-provenance.json) · [Regional support](results/minimum-fit-2026-11-25/support/README.md) · [Whole-area sensitivity](results/minimum-fit-2026-11-25/nested-whole-area/README.md)

[Previous maps and geological assessment](previous-minimum-treatment.md) · [Change at all distances across eruptions](../../comparisons/reported-minima/README.md)
