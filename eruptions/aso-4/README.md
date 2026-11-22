# Aso-4 fallout reconstruction

**Geographic reach of nested isopachs.** Limiting the lateral and down-range reach of each positive site gives 652.5 km³ of bulk fallout (conditional 95% interval 249.5–9,990.1), compared with 2,513.9 km³ under the whole-area continuation. The western sector has no measured thickness and its median 1 cm footprint changes from 6.32 million to 48,000 km². On a separate test, 28 actual submarine cores were better predicted by the broader fit. [Paired maps, volumes and checks](results/nested-reach/README.md) · [Full-compass isopachs](results/nested-reach/compass.png)

The accepted dataset contains 83 exact measurements, 3 lower bounds and 1 upper bound. The regional fits share an observation hull of 3,482,991 km². [Accepted thickness CSV](../../data/eruptions/Aso-4_reviewed.csv) · [Row-level source audit](../../data/audits/Aso-4/README.md)

The median modeled area above 1 mm covers 44.3% of the Earth. With the farthest fifth of sites withheld, its predictive score worsens from 1.757 to 1.836. That test and the inferred footprint make the 2,514 km³ median a highly extrapolated scenario. The compass-sector test shows what happens when the unknown fan can turn west or south. [Distal and angular sensitivities](results/nested-isopachs/bearing-study/README.md) · [Known-volume and distal holdout tests](../../comparisons/nested-isopachs/README.md)

The 1 cm marine bed at MR98-05 St.6PC sets the outermost sampled rank in 70% of draws. Withholding that bed gives 2,439 km³. The large volume persists because the fitted contour shape spreads the sampled centimetre-scale thicknesses over a much broader area. The western sector has no exact measurements and receives a median 12.5% of the modeled volume. [Source-to-core correspondence](../../data/audits/Aso-4/record-review.csv) · [Refit without the controlling core](results/nested-isopachs/edge-deletion/README.md)

Takarada and Hoshizumi reconstructed 590–920 km³ of bulk Aso-4 fallout from manual isopachs on 71 sites. Their reconstruction shares source evidence with this audit. It provides a geological comparison and supplied no prior target. [Original fallout-volume reconstruction](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.00170/full)

## Isopach fits

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at Aso-4](results/four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, solid white segments connect thicknesses bracketed by measured sites in a supported triangle. White dashes on the global maps show conditional isopachs. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **202.65** | 155.87–266.97 | 1.514 | 94.0% |
| Cubic spline | Hull mean **248.15** | 168.43–452.02 | 1.662 | 90.4% |
| [Directional ensemble · uncertain fan direction](results/directional/README.md) | Whole-Earth median map **659.1** · volume median 916.3 | 297.9–3,511.5 | 1.502 | 90.4% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **1,069.7** | 402.9–27,836.6 | 1.632 | 95.2% |
| [Bayesian nested isopachs](results/nested-isopachs/README.md) | Whole-Earth volume median **2,513.9** | 648.8–15,969.9 | 1.540 † | 90.4% |

[Nested median isopachs and the area–thickness curve](results/nested-isopachs/isopachs.png) · [Nested whole-Earth map](results/nested-isopachs/world.png) · [Nested sensitivity in poorly sampled bearings](results/nested-isopachs/bearing-study/README.md) · [Uncertain gap amount](results/directional/sensitivity-study/README.md) · [Soft source displacement](results/spherical-mixture/sensitivity-study/README.md)

The nested estimate puts a median 1.2% of the fallout volume inside the retained exact-site triangles and 55.1% beyond the farthest exact measurement. 3/4 geographic refits passed the independent-population check.

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **3.8%** of its volume. The directional volume interval comes from complete simulated deposits. The spherical mixture reports the median integrated volume across broad and uncertain fan continuations. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the two whole-Earth maps</summary>

| Uncertain fan direction | Spherical mixture |
|---|---|
| ![Uncertain-fan-direction median thickness over the Earth for Aso-4](results/directional/azimuth/world.png) | ![Spherical-mixture median thickness over the Earth for Aso-4](results/spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>

With the same observations and structure weights, fixing the fan directions gives a median-map volume of **779.7 km³**.

<details>
<summary>View the directional deposit extent and source area</summary>

![Principal uncertain-fan-direction median isopachs for Aso-4](results/directional/azimuth/isopachs.png)

</details>

## Evidence and limitations

The reviewed input corrects two cruise locations and retains the Biwa source conflict and the Hokkaido erosion interpretation. The nearest accepted site is 88 km from the vent. The southeast, south and west have two, two and zero exact measurements. The large spherical-mixture total remains questioned. In the earlier broad spherical fit, omitting all 28 source-labelled marine records raised its median to about 3,184 km³. A regional deletion also makes the spline partial volume unstable. [Source, geometry and influence experiments](../../comparisons/aso-4-audit/README.md)

The spherical 10 cm footprint contracts to the south, southeast and west. The displaced-kernel map still places its near-source grid maximum southeast of the volcano, away from the closest measurements. Its location and the large upper volume tail depend on extrapolation. The south and southwest 10 cm areas retain numerical warnings. [Near-source spherical map](results/spherical-mixture/source.png) · [Matched isopach areas and known-deposit tests](../../comparisons/spherical-mixture/README.md#where-the-map-changes)

The standard local-support footprint covers 711,348 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
