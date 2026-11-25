# B-Tm · fixed-direction ensemble

**31.1 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **23.5–50.2 km³**.

The measured triangles contain **4.7 km³** (15.1%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 54.1 cm. The largest median thickness on the two display grids is 55.5 cm at a source distance of 28.7 km. The evidence panel contains 4 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 0 | 0.0 | 0.0–0.2 |
| NE | 2 | 1 | 1.5 | 0.1–9.6 |
| E | 64 | 0 | 27.8 | 21.9–37.5 |
| SE | 0 | 1 | 1.6 | 0.6–7.0 |
| S | 0 | 0 | 0.0 | 0.0–0.5 |
| SW | 0 | 0 | 0.0 | 0.0–0.1 |
| W | 0 | 1 | 0.1 | 0.0–0.5 |
| NW | 0 | 1 | 0.0 | 0.0–0.1 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 4.7 | 3.9–6.4 |
| Transition at triangle edges | 3.6 | 3.0–4.6 |
| Proximal background | 0.0 | 0.0–0.0 |
| Background between sampled radii | 19.9 | 12.4–35.8 |
| Distal background | 3.0 | 1.5–12.3 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.7% | 22.3–44.3 | passed |
| 45° | Absent from this structure | 16.6% | 27.1–45.8 | passed |
| 90° | Absent from this structure | 11.9% | 33.8–55.7 | passed |
| 22.5° | Possible | 14.9% | 22.0–34.7 | passed |
| 45° | Possible | 22.9% | 26.5–40.7 | passed |
| 90° | Possible | 18.0% | 33.3–52.7 | passed |

Equal weights give a median-map volume of **31.1 km³**. Restricting to the three structures without an extra fan gives **31.1 km³**. Allowing an extra fan in all three gives **31.2 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 71/71 | 1.318 | 1.318 | 1.314 | 94.0% |
| Quadrants | 71/71 | 2.040 | 1.964 | 4.341 | 92.5% |

Geographic sampling warnings occurred in folds 0. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. Unscored folds were 2 (empty).

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 16,384 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.5%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.06%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
