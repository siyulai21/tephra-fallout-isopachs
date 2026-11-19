# B-Tm directional ensemble

[Measured and extrapolated directional contours](azimuth/README.md) shows which median contour segments the measurements bracket, how an unsampled fan changes their reach, and how the alternatives predict withheld observations.

[Directional reach and archive evidence](bearing/README.md) gives median isopachs, sector volumes and the sensitivity to unsampled directions. The six-structure calculation below includes a possible additional fan in the largest sampling gap.

**54.0 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **73.5 km³**, with a conditional 95% interval of **31.8–1,211.9 km³**.

The measured triangles contain **4.4 km³** (8.2%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](ensemble/isopachs.png)

The star marks the supplied vent. Its median thickness is 107.6 cm. The largest median thickness on the two display grids is 107.6 cm at a source distance of 0.0 km. The evidence panel contains 5 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](ensemble/local.png)

![Median thickness over the Earth](ensemble/world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 0 | 1 | 1.6 | 2.0 [0.1–10.9] |
| NE | 0 | 2 | 3.9 | 5.1 [0.5–19.9] |
| E | 64 | 0 | 30.2 | 32.7 [22.9–65.9] |
| SE | 0 | 1 | 4.2 | 4.9 [1.4–14.1] |
| S | 0 | 0 | 1.8 | 2.1 [0.2–11.1] |
| SW | 0 | 0 | 2.7 | 3.4 [0.2–147.7] |
| W | 0 | 0 | 6.7 | 7.6 [0.2–859.2] |
| NW | 0 | 0 | 2.7 | 3.4 [0.2–138.7] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 4.4 | 4.7 [3.7–6.0] |
| Transition at triangle edges | 3.6 | 3.8 [3.1–4.9] |
| Proximal background | 0.6 | 0.6 [0.3–1.5] |
| Background between sampled radii | 38.7 | 49.0 [16.9–287.1] |
| Distal background | 6.7 | 12.3 [2.7–931.2] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.7% | 50.5 [26.8–126.7] | passed |
| 45° | Absent from this structure | 16.6% | 55.5 [33.7–134.7] | passed |
| 90° | Absent from this structure | 11.9% | 60.1 [44.8–120.0] | passed |
| 22.5° | Possible | 14.9% | 83.0 [32.1–726.1] | passed |
| 45° | Possible | 22.9% | 100.8 [40.4–1,548.8] | passed |
| 90° | Possible | 18.0% | 142.2 [54.4–3,836.4] | passed |

Equal weights give a median-map volume of **51.9 km³**. Restricting to the three structures without an extra fan gives **46.5 km³**. Allowing an extra fan in all three gives **79.3 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 68/68 | 1.304 | 1.302 | 1.425 | 93.8% |
| Quadrants | 68/68 | 1.428 | 1.432 | 1.448 | 90.6% |

Geographic sampling warnings occurred in folds 0. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 1. Unscored folds were 2 (empty), 3 (empty).

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.6%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](ensemble/report.json) · [Complete contour decisions](ensemble/complete-contour-audit.json) · [All fragment decisions](ensemble/fragment-decisions.csv) · [Mapped fragment vertices](ensemble/supported-fragments.csv) · [Held-out predictions](ensemble/heldout.csv)

[Reconstruct the median field](ensemble/map-fields.npz) · [Map grids and pointwise quantiles](ensemble/median-grids.npz) · [Joint volume by chain, support class and compass sector](ensemble/coherent-mass.npz) · [Checksums](ensemble/artifacts.json)

[Single-structure sensitivity study and unmeasured proximal cap](single-structure.md)
