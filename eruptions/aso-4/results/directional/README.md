# Aso-4 directional ensemble

**740.0 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **882.7 km³**, with a conditional 95% interval of **397.5–2,931.7 km³**.

The measured triangles contain **24.8 km³** (3.3%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](ensemble/isopachs.png)

The star marks the supplied vent. Its median thickness is 39.8 cm. The largest median thickness on the two display grids is 79.3 cm at a source distance of 40.0 km. The evidence panel contains 10 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](ensemble/local.png)

![Median thickness over the Earth](ensemble/world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 8 | 0 | 50.2 | 56.7 [11.4–308.8] |
| NE | 60 | 4 | 357.7 | 376.0 [168.0–944.7] |
| E | 11 | 0 | 243.2 | 249.9 [92.5–1,262.9] |
| SE | 2 | 0 | 50.5 | 57.6 [13.3–378.8] |
| S | 2 | 0 | 8.3 | 9.1 [4.5–190.1] |
| SW | 0 | 0 | 12.8 | 14.1 [4.7–401.7] |
| W | 0 | 0 | 9.5 | 11.1 [4.6–249.1] |
| NW | 0 | 0 | 7.8 | 8.8 [4.3–184.0] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 24.8 | 25.3 [20.4–32.0] |
| Transition at triangle edges | 28.2 | 30.3 [24.7–38.8] |
| Proximal background | 4.6 | 4.7 [2.3–9.5] |
| Background between sampled radii | 238.9 | 262.7 [169.3–540.3] |
| Distal background | 443.5 | 559.5 [146.4–2,384.5] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.6% | 756.9 [350.0–2,453.1] | passed |
| 45° | Absent from this structure | 17.2% | 743.5 [391.5–2,030.2] | passed |
| 90° | Absent from this structure | 11.4% | 1,214.2 [477.5–3,171.5] | passed |
| 22.5° | Possible | 14.9% | 827.3 [382.1–2,700.5] | passed |
| 45° | Possible | 23.9% | 827.0 [411.6–2,780.3] | passed |
| 90° | Possible | 17.0% | 1,351.4 [510.5–3,511.7] | passed |

Equal weights give a median-map volume of **753.6 km³**. Restricting to the three structures without an extra fan gives **710.0 km³**. Allowing an extra fan in all three gives **776.2 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 87/87 | 1.507 | 1.501 | 1.555 | 86.7% |
| Quadrants | 87/87 | 1.325 | 1.316 | 1.550 | 90.4% |

Geographic sampling warnings occurred in zero folds. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. Unscored folds were 2 (empty).

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.5%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](ensemble/report.json) · [Complete contour decisions](ensemble/complete-contour-audit.json) · [All fragment decisions](ensemble/fragment-decisions.csv) · [Mapped fragment vertices](ensemble/supported-fragments.csv) · [Held-out predictions](ensemble/heldout.csv)

[Reconstruct the median field](ensemble/map-fields.npz) · [Map grids and pointwise quantiles](ensemble/median-grids.npz) · [Joint volume by chain, support class and compass sector](ensemble/coherent-mass.npz) · [Checksums](ensemble/artifacts.json)

[Single-structure sensitivity study and unmeasured proximal cap](single-structure.md)
