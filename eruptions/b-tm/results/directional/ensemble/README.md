# B-Tm directional ensemble

**59.3 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. Across complete simulated deposits the median volume is **71.0 km³**, with a conditional 95% interval of **42.2–410.0 km³**.

The measured triangles contain **2.2 km³** (3.6%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 130.2 cm. The largest median thickness on the two display grids is 130.2 cm at a source distance of 0.0 km. The evidence panel contains 20 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 0 | 0.8 | 0.8 [0.3–2.5] |
| NE | 5 | 0 | 1.5 | 1.6 [0.8–3.5] |
| E | 75 | 0 | 44.7 | 45.6 [29.2–109.2] |
| SE | 2 | 0 | 7.5 | 8.2 [3.3–20.6] |
| S | 1 | 0 | 1.0 | 1.1 [0.4–3.4] |
| SW | 0 | 0 | 1.1 | 1.3 [0.3–65.0] |
| W | 0 | 0 | 1.9 | 2.3 [0.3–253.8] |
| NW | 0 | 0 | 0.9 | 1.1 [0.3–21.6] |

| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |
|---|---:|---:|
| Measured triangles | 2.2 | 2.2 [1.8–2.7] |
| Transition at triangle edges | 4.6 | 4.9 [4.0–6.0] |
| Proximal background | 0.0 | 0.0 [0.0–0.0] |
| Background between sampled radii | 26.2 | 29.3 [19.2–96.5] |
| Distal background | 26.3 | 32.5 [11.8–310.4] |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.7% | 64.5 [37.8–182.5] | passed |
| 45° | Absent from this structure | 16.6% | 61.6 [39.6–109.0] | passed |
| 90° | Absent from this structure | 11.9% | 59.8 [46.3–89.3] | passed |
| 22.5° | Possible | 14.9% | 83.4 [43.1–387.1] | passed |
| 45° | Possible | 22.9% | 81.8 [46.1–556.1] | passed |
| 90° | Possible | 18.0% | 83.0 [51.8–863.7] | passed |

Equal weights give a median-map volume of **58.8 km³**. Restricting to the three structures without an extra fan gives **57.6 km³**. Allowing an extra fan in all three gives **68.7 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 84/84 | 1.295 | 1.297 | 1.316 | 95.2% |
| Quadrants | 84/84 | 1.989 | 1.961 | 3.191 | 88.1% |

Geographic sampling warnings occurred in zero folds. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. Unscored folds were 2 (empty), 3 (empty).

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 16,384 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.1%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.02%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
