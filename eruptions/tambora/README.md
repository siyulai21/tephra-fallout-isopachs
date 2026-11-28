# Tambora fallout reconstruction

[Principal input maps and numbered detail plates](observations/README.md)

The accepted dataset contains 26 exact measurements. The regional fits share an observation hull of 2,980,287 km². [Accepted thickness CSV](../../data/eruptions/Tambora_tephra.csv) · [Input provenance](../../data/eruptions/README.md)

## Volumes and geographic prediction

Each estimate integrates the median thickness on its map. The conditional range integrates the pointwise 25th and 75th percentiles of thickness over the same area. The 50% probability refers to thickness at each location. A single probability for total volume does not follow from these endpoints.

| Method | Integration area | Median map (km³) | Q25–Q75 thickness maps (km³) | Held-out score ↓ | 95% predictive coverage |
|---|---|---:|---:|---:|---:|
| GP ensemble | observation hull | **91.1** | 71.4–117 | 1.620 | 96.2% |
| Cubic spline | observation hull | **93.5** | 73.5–123 | 1.548 | 96.2% |
| [Multiscale directional ensemble](results/directional/README.md) | whole Earth | **135** | 90.9–224 | 1.437 | 100.0% |
| [Spherical mixture](results/spherical-mixture/README.md) | whole Earth | **117** | 82.3–176 | 1.590 | 100.0% |
| [Bayesian nested footprints](results/nested-isopachs/README.md) | whole Earth | **131** | 87.5–215 | 1.520 | 96.2% |
| [Wrapped angular covariance](results/wrapped-angular-covariance/README.md) | whole Earth | **119** | 88.2–164 | 1.362 | 95.8% |

Lower scores indicate better geographic prediction. Coverage counts exact withheld measurements. A dagger marks a fit, integration or validation warning.

## Full deposits and near-vent detail

Each full map follows the thickest power-of-two contour enclosing at least 95% of the fitted locations, with a 1 cm minimum. The title gives the actual count, including when that minimum prevents 95% coverage. Global volumes include the thinner tail. The innermost line is the thickest doubling level enclosing an accepted location. The near-vent window expands from the proximal sites until eligible contours are visible. The curve sums all patches at each displayed thickness.

![Tambora · Multiscale directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/directional/deposit.png)

![Tambora · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/spherical-mixture/deposit.png)

![Tambora · Bayesian nested footprints. Full deposit, near-vent detail and thickness versus square root of area.](results/nested-isopachs/deposit.png)

![Tambora · Wrapped angular covariance. Full deposit, near-vent detail and thickness versus square root of area.](results/wrapped-angular-covariance/deposit.png)

## Inside the observation hull

![Tambora · GP ensemble and cubic spline on a common scale and the full observation hull](results/isopachs.png)

The GP and spline integrate over the dashed hull. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/README.md)

[Wrapped angular covariance at three scales](results/wrapped-angular-covariance/three-scales.png) · [Geological review and matched validation](../../comparisons/wrapped-angular-covariance/evidence.md)

The current global fits use the same audited rows. [What changed in the maps, volumes and validation](../../comparisons/global-extent/README.md) · [Continuation rules](../../docs/methods/finite-spread.md)

## Evidence and limitations

The calculation uses the supplied acceptance flags. An independent row-by-row source review is still outstanding. The median-map integration checks passed, but several held-out fits also carry warnings.

The standard local-support footprint covers 49,323 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)

In the [earlier fitted comparison](../../archive/2026-11-22-global-extent/README.md), both geographic and whole-quadrant holdouts predict the real observations better under the broader continuation. [Whole-area continuation sensitivity](results/nested-whole-area/README.md)
