# Sakurajima Satsuma (P14) · fixed-direction ensemble

**8.6 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **4.5–73.6 km³**.

The measured triangles contain **0.2 km³** (2.1%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 104.9 cm. The largest median thickness on the two display grids is 104.9 cm at a source distance of 0.0 km. The evidence panel contains 22 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 2 | 2 | 0.5 | 0.3–3.7 |
| NE | 6 | 1 | 0.3 | 0.2–2.0 |
| E | 20 | 1 | 0.7 | 0.4–5.4 |
| SE | 8 | 0 | 2.0 | 0.7–20.9 |
| S | 6 | 2 | 3.1 | 1.1–29.9 |
| SW | 3 | 1 | 1.0 | 0.4–10.1 |
| W | 3 | 0 | 0.5 | 0.3–2.8 |
| NW | 2 | 0 | 0.4 | 0.1–3.8 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 0.2 | 0.2–0.2 |
| Transition at triangle edges | 0.4 | 0.4–0.5 |
| Proximal background | 0.2 | 0.1–0.3 |
| Background between sampled radii | 4.0 | 2.7–6.1 |
| Distal background | 3.8 | 0.8–67.4 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 4.0–50.7 | passed |
| 45° | Absent from this structure | 16.8% | 4.9–62.2 | passed |
| 90° | Absent from this structure | 11.9% | 6.3–120.0 | passed |
| 22.5° | Possible | 14.8% | 4.1–50.8 | passed |
| 45° | Possible | 23.0% | 5.0–62.9 | passed |
| 90° | Possible | 18.0% | 6.5–98.7 | passed |

Equal weights give a median-map volume of **8.6 km³**. Restricting to the three structures without an extra fan gives **8.3 km³**. Allowing an extra fan in all three gives **8.8 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 57/57 | 1.069 | 1.071 | 1.264 | 92.0% |
| Quadrants | 57/57 | 1.283 | 1.300 | 1.629 | 90.0% |

Geographic sampling warnings occurred in folds 2. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. All four folds produced predictions.

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 1.6%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
