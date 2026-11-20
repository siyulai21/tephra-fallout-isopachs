# B-Tm · fixed-direction ensemble

**33.9 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **36.7 km³**, with a conditional 95% interval of **23.6–63.0 km³**.

The measured triangles contain **4.7 km³** (13.8%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 202.8 cm. The largest median thickness on the two display grids is 202.8 cm at a source distance of 0.0 km. The evidence panel contains 3 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 0 | 1 | 0.1 | 0.1 [0.0–0.3] |
| NE | 0 | 3 | 1.4 | 1.5 [0.1–16.3] |
| E | 64 | 0 | 30.3 | 30.9 [21.7–41.9] |
| SE | 0 | 1 | 1.8 | 2.1 [0.7–8.7] |
| S | 0 | 0 | 0.1 | 0.1 [0.0–0.7] |
| SW | 0 | 0 | 0.1 | 0.1 [0.0–0.3] |
| W | 0 | 1 | 0.1 | 0.1 [0.0–0.6] |
| NW | 0 | 1 | 0.1 | 0.1 [0.0–0.3] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 4.7 | 5.0 [3.9–6.5] |
| Transition at triangle edges | 3.6 | 3.7 [3.0–4.8] |
| Proximal background | 0.4 | 0.4 [0.1–1.9] |
| Background between sampled radii | 22.0 | 22.8 [12.0–47.5] |
| Distal background | 3.3 | 3.8 [1.5–12.2] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.7% | 28.0 [22.3–37.8] | passed |
| 45° | Absent from this structure | 16.6% | 37.5 [29.5–49.7] | passed |
| 90° | Absent from this structure | 11.9% | 53.5 [41.7–69.8] | passed |
| 22.5° | Possible | 14.9% | 27.3 [22.0–35.2] | passed |
| 45° | Possible | 22.9% | 36.1 [28.7–46.3] | passed |
| 90° | Possible | 18.0% | 51.7 [40.8–67.2] | passed |

Equal weights give a median-map volume of **33.9 km³**. Restricting to the three structures without an extra fan gives **34.0 km³**. Allowing an extra fan in all three gives **33.9 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 71/71 | 1.266 | 1.267 | 1.274 | 93.8% |
| Quadrants | 71/71 | 2.663 | 2.563 | 4.508 | 87.5% |

Geographic sampling warnings occurred in folds 0. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. Unscored folds were 2 (empty).

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 16,384 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.3%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.03%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
