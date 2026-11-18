# Mazama directional ensemble

**179.9 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **224.3 km³**, with a conditional 95% interval of **149.1–1,371.9 km³**.

The measured triangles contain **26.2 km³** (14.6%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 418.2 cm. The largest median thickness on the two display grids is 560.3 cm at a source distance of 24.7 km. The evidence panel contains 15 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 21 | 0 | 33.6 | 34.9 [27.5–45.7] |
| NE | 19 | 0 | 92.1 | 101.1 [58.6–273.2] |
| E | 3 | 0 | 21.7 | 25.2 [11.6–74.1] |
| SE | 0 | 0 | 4.6 | 5.2 [2.1–13.8] |
| S | 0 | 0 | 5.0 | 6.1 [1.8–278.2] |
| SW | 0 | 0 | 7.7 | 9.5 [2.2–810.8] |
| W | 0 | 0 | 6.5 | 7.2 [2.7–69.6] |
| NW | 2 | 0 | 8.8 | 9.2 [4.5–17.6] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 26.2 | 27.0 [21.0–34.9] |
| Transition at triangle edges | 61.5 | 62.9 [45.1–89.2] |
| Proximal background | 0.0 | 0.0 [0.0–0.0] |
| Background between sampled radii | 78.6 | 101.2 [62.4–538.6] |
| Distal background | 13.6 | 24.8 [3.6–766.0] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 16.1% | 216.5 [153.2–464.8] | passed |
| 45° | Absent from this structure | 15.7% | 203.3 [150.6–370.2] | passed |
| 90° | Absent from this structure | 12.3% | 179.8 [133.0–258.8] | passed |
| 22.5° | Possible | 15.7% | 253.8 [165.7–1,047.4] | passed |
| 45° | Possible | 21.2% | 254.7 [168.1–1,741.3] | passed |
| 90° | Possible | 18.9% | 261.0 [154.7–3,025.1] | passed |

Equal weights give a median-map volume of **177.5 km³**. Restricting to the three structures without an extra fan gives **178.7 km³**. Allowing an extra fan in all three gives **201.1 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 45/45 | 1.407 | 1.413 | 1.451 | 95.6% |
| Quadrants | 45/45 | 1.990 | 1.995 | 2.070 | 68.9% |

Geographic sampling warnings occurred in zero folds. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 0. Unscored folds were 2 (empty).

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 16,384 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.0%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.00%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
