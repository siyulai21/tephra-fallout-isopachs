# Ilopango TBJ (Unit G) · fixed-direction ensemble

**47.3 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **64.2 km³**, with a conditional 95% interval of **33.5–368.2 km³**.

The measured triangles contain **1.6 km³** (3.4%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 504.9 cm. The largest median thickness on the two display grids is 504.9 cm at a source distance of 0.0 km. The evidence panel contains 21 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 3 | 0 | 1.3 | 1.3 [0.9–16.7] |
| NE | 2 | 1 | 2.0 | 2.4 [0.9–41.0] |
| E | 9 | 1 | 6.1 | 8.6 [2.4–118.2] |
| SE | 5 | 1 | 13.0 | 15.9 [5.6–117.5] |
| S | 12 | 0 | 9.9 | 10.9 [5.8–65.5] |
| SW | 2 | 0 | 6.2 | 6.5 [3.4–34.5] |
| W | 12 | 1 | 4.8 | 5.0 [2.7–23.8] |
| NW | 7 | 0 | 3.9 | 4.3 [2.2–27.6] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 1.6 | 1.6 [1.3–1.9] |
| Transition at triangle edges | 3.1 | 3.2 [2.7–3.8] |
| Proximal background | 0.6 | 0.6 [0.3–1.4] |
| Background between sampled radii | 24.3 | 26.8 [18.5–39.5] |
| Distal background | 17.8 | 31.3 [6.4–329.5] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 56.2 [29.9–341.5] | passed |
| 45° | Absent from this structure | 16.8% | 69.9 [35.8–449.3] | passed |
| 90° | Absent from this structure | 11.9% | 88.9 [40.2–525.3] | passed |
| 22.5° | Possible | 14.8% | 60.2 [31.0–388.1] | passed |
| 45° | Possible | 23.0% | 63.1 [36.5–239.4] | passed |
| 90° | Possible | 18.0% | 59.7 [35.6–295.5] | passed |

Equal weights give a median-map volume of **47.6 km³**. Restricting to the three structures without an extra fan gives **49.9 km³**. Allowing an extra fan in all three gives **46.5 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 56/56 | 1.501 | 1.509 | 1.728 | 90.4% |
| Quadrants | 56/56 | 1.459 | 1.477 | 1.860 | 96.2% |

Geographic sampling warnings occurred in zero folds. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 1. All four folds produced predictions.

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.5%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
