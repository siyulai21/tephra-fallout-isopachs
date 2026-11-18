# Tambora directional ensemble

**105.8 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **119.6 km³**, with a conditional 95% interval of **78.6–184.0 km³**.

The measured triangles contain **10.7 km³** (10.1%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 103.8 cm. The largest median thickness on the two display grids is 106.4 cm at a source distance of 11.5 km. The evidence panel contains 10 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 2 | 0 | 14.6 | 15.4 [7.4–26.3] |
| NE | 3 | 0 | 12.1 | 12.7 [7.1–20.8] |
| E | 4 | 0 | 11.5 | 12.4 [6.8–20.7] |
| SE | 2 | 0 | 7.1 | 7.5 [2.3–21.2] |
| S | 1 | 0 | 10.8 | 11.6 [3.4–45.4] |
| SW | 1 | 0 | 11.2 | 12.1 [4.2–26.6] |
| W | 13 | 0 | 24.6 | 26.4 [16.9–73.3] |
| NW | 0 | 0 | 13.9 | 14.9 [1.5–30.4] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 10.7 | 11.0 [8.5–14.5] |
| Transition at triangle edges | 30.6 | 30.8 [23.6–41.3] |
| Proximal background | 0.0 | 0.0 [0.0–0.0] |
| Background between sampled radii | 63.9 | 74.6 [40.3–125.5] |
| Distal background | 0.6 | 1.6 [0.3–23.1] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 13.6% | 112.1 [72.7–170.3] | **review** |
| 45° | Absent from this structure | 16.5% | 112.1 [77.2–172.9] | **review** |
| 90° | Absent from this structure | 14.4% | 124.3 [92.7–178.7] | passed |
| 22.5° | Possible | 13.3% | 120.9 [77.1–219.0] | **review** |
| 45° | Possible | 22.4% | 116.1 [77.6–193.9] | **review** |
| 90° | Possible | 19.8% | 125.8 [94.2–185.0] | passed |

Equal weights give a median-map volume of **104.7 km³**. Restricting to the three structures without an extra fan gives **103.2 km³**. Allowing an extra fan in all three gives **107.6 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 26/26 | 1.964 | 1.979 | 1.953 | 84.6% |
| Quadrants | 26/26 | 1.794 | 1.820 | 1.959 | 96.2% |

Geographic sampling warnings occurred in folds 0, 1, 3. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 0, 1, 2, 3. All four folds produced predictions.

Full-data structures requiring numerical review were narrow fans, reference, narrow missing fan, possible missing fan. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 1.0%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
