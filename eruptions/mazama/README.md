# Mazama fallout reconstruction

The accepted dataset contains 45 exact measurements. The regional fits share an observation hull of 930,447 km². [Accepted thickness CSV](../../data/eruptions/Mazama_tephra.csv) · [Input provenance](../../data/eruptions/README.md)

## Isopach fits

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at Mazama](results/four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, solid white segments connect thicknesses bracketed by measured sites in a supported triangle. White dashes on the global maps show conditional isopachs. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **141.64** | 109.26–179.21 | 1.272 | 95.6% |
| Cubic spline | Hull mean **132.61** | 120.25–151.97 | 1.971 | 82.2% |
| [Directional ensemble · uncertain fan direction](results/directional/README.md) | Whole-Earth median map **166.1** · volume median 213.5 | 138.6–1,265.1 | 1.408 | 95.6% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **212.9** | 140.1–2,614.3 | 1.561 † | 88.9% |
| [Bayesian nested isopachs](results/nested-isopachs/README.md) | Whole-Earth volume median **411.2** | 162.2–1,663.6 | 1.503 | 91.1% |

[Nested median isopachs and the area–thickness curve](results/nested-isopachs/isopachs.png) · [Nested whole-Earth map](results/nested-isopachs/world.png) · [Nested sensitivity in poorly sampled bearings](results/nested-isopachs/bearing-study/README.md) · [Uncertain gap amount](results/directional/sensitivity-study/README.md) · [Soft source displacement](results/spherical-mixture/sensitivity-study/README.md)

The nested estimate puts a median 5.4% of the fallout volume inside the retained exact-site triangles and 15.3% beyond the farthest exact measurement. 4/4 geographic refits passed the independent-population check.

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval. † A geographic fit had a numerical warning or incomplete scoring.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **15.7%** of its volume. The directional volume interval comes from complete simulated deposits. The spherical mixture reports the median integrated volume across broad and uncertain fan continuations. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the two whole-Earth maps</summary>

| Uncertain fan direction | Spherical mixture |
|---|---|
| ![Uncertain-fan-direction median thickness over the Earth for Mazama](results/directional/azimuth/world.png) | ![Spherical-mixture median thickness over the Earth for Mazama](results/spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>

With the same observations and structure weights, fixing the fan directions gives a median-map volume of **177.0 km³**.

<details>
<summary>View the directional deposit extent and source area</summary>

![Principal uncertain-fan-direction median isopachs for Mazama](results/directional/azimuth/isopachs.png)

</details>

## Evidence and limitations

The calculation uses the supplied acceptance flags. An independent row-by-row source review is still outstanding. Some spherical-mixture prior refits carry numerical warnings, and the broad upper interval reflects unsampled fallout.

The standard local-support footprint covers 172,792 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
