# Sakurajima Satsuma (P14) · fixed-direction ensemble

**8.3 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **4.4–87.7 km³**.

The measured triangles contain **0.2 km³** (2.2%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 87.8 cm. The largest median thickness on the two display grids is 87.8 cm at a source distance of 0.0 km. The evidence panel contains 25 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 4 | 0 | 0.4 | 0.2–3.6 |
| NE | 6 | 1 | 0.4 | 0.2–2.9 |
| E | 21 | 0 | 0.7 | 0.4–7.3 |
| SE | 8 | 0 | 2.0 | 0.7–23.3 |
| S | 8 | 0 | 2.8 | 1.0–31.8 |
| SW | 3 | 1 | 1.0 | 0.4–12.9 |
| W | 3 | 0 | 0.6 | 0.3–4.4 |
| NW | 2 | 0 | 0.4 | 0.1–4.9 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 0.2 | 0.1–0.2 |
| Transition at triangle edges | 0.4 | 0.4–0.5 |
| Proximal background | 0.1 | 0.1–0.3 |
| Background between sampled radii | 3.8 | 2.6–5.7 |
| Distal background | 3.8 | 0.8–82.0 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 3.9–63.2 | passed |
| 45° | Absent from this structure | 16.8% | 4.7–82.5 | passed |
| 90° | Absent from this structure | 11.9% | 5.8–120.0 | passed |
| 22.5° | Possible | 14.8% | 4.0–72.4 | passed |
| 45° | Possible | 23.0% | 4.9–87.3 | passed |
| 90° | Possible | 18.0% | 5.8–100.3 | passed |

Equal weights give a median-map volume of **8.3 km³**. Restricting to the three structures without an extra fan gives **8.0 km³**. Allowing an extra fan in all three gives **8.6 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 57/57 | 1.127 | 1.130 | 1.289 | 94.5% |
| Quadrants | 57/57 | 1.333 | 1.345 | 1.602 | 85.5% |

Geographic sampling warnings occurred in folds 2. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. All four folds produced predictions.

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.7%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.00%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
