# YTT directional ensemble

[Measured and extrapolated directional contours](azimuth/README.md) shows which median contour segments the measurements bracket, how an unsampled fan changes their reach, and how the alternatives predict withheld observations.

**985.7 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **1,274.1 km³**, with a conditional 95% interval of **765.6–3,514.0 km³**.

The measured triangles contain **193.8 km³** (19.7%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](ensemble/isopachs.png)

The star marks the supplied vent. Its median thickness is 27.2 cm. The largest median thickness on the two display grids is 34.7 cm at a source distance of 487.7 km. The evidence panel contains 20 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](ensemble/local.png)

![Median thickness over the Earth](ensemble/world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 0 | 0 | 95.4 | 158.9 [19.0–328.0] |
| NE | 5 | 0 | 78.7 | 81.9 [43.5–188.6] |
| E | 1 | 0 | 84.2 | 90.5 [17.6–813.8] |
| SE | 2 | 0 | 75.2 | 84.4 [19.1–1,619.6] |
| S | 1 | 0 | 40.6 | 46.5 [15.0–227.9] |
| SW | 0 | 0 | 14.4 | 20.0 [5.4–107.5] |
| W | 11 | 0 | 256.6 | 302.4 [205.1–491.2] |
| NW | 26 | 0 | 340.4 | 367.8 [272.4–633.9] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 193.8 | 196.5 [144.3–276.5] |
| Transition at triangle edges | 119.7 | 124.6 [90.5–175.8] |
| Proximal background | 3.0 | 3.1 [0.3–6.2] |
| Background between sampled radii | 659.5 | 903.9 [464.2–2,440.8] |
| Distal background | 9.7 | 23.7 [2.8–768.7] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 16.2% | 909.3 [695.6–1,445.1] | passed |
| 45° | Absent from this structure | 17.8% | 1,170.0 [848.5–1,926.9] | passed |
| 90° | Absent from this structure | 14.3% | 1,372.6 [1,080.7–1,906.0] | passed |
| 22.5° | Possible | 15.4% | 1,081.4 [744.6–4,020.4] | passed |
| 45° | Possible | 18.3% | 1,541.3 [943.7–7,366.2] | passed |
| 90° | Possible | 18.0% | 1,412.5 [1,114.2–2,042.3] | passed |

Equal weights give a median-map volume of **982.0 km³**. Restricting to the three structures without an extra fan gives **932.0 km³**. Allowing an extra fan in all three gives **1,058.7 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 46/46 | 1.777 | 1.782 | 1.811 | 93.5% |
| Quadrants | 46/46 | 3.069 | 3.085 | 3.782 | 37.0% |

Geographic sampling warnings occurred in folds 1, 3. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. All four folds produced predictions.

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 16,384 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.2%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.03%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](ensemble/report.json) · [Complete contour decisions](ensemble/complete-contour-audit.json) · [All fragment decisions](ensemble/fragment-decisions.csv) · [Mapped fragment vertices](ensemble/supported-fragments.csv) · [Held-out predictions](ensemble/heldout.csv)

[Reconstruct the median field](ensemble/map-fields.npz) · [Map grids and pointwise quantiles](ensemble/median-grids.npz) · [Joint volume by chain, support class and compass sector](ensemble/coherent-mass.npz) · [Checksums](ensemble/artifacts.json)

[Single-structure sensitivity study and unmeasured proximal cap](single-structure.md)
