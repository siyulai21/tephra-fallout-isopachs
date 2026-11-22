# Pinatubo 1991 (Layer C) · fixed-direction ensemble

**6.9 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **5.8–10.8 km³**.

The measured triangles contain **0.6 km³** (8.4%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 89.6 cm. The largest median thickness on the two display grids is 89.6 cm at a source distance of 0.0 km. The evidence panel contains 50 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 2 | 0.0 | 0.0–0.2 |
| NE | 12 | 2 | 0.0 | 0.0–0.1 |
| E | 30 | 9 | 0.1 | 0.1–0.1 |
| SE | 19 | 10 | 0.1 | 0.1–0.1 |
| S | 10 | 7 | 0.7 | 0.3–1.9 |
| SW | 33 | 8 | 2.3 | 1.6–5.1 |
| W | 24 | 12 | 3.4 | 3.0–4.2 |
| NW | 11 | 4 | 0.2 | 0.1–1.0 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 0.6 | 0.5–0.7 |
| Transition at triangle edges | 0.2 | 0.2–0.2 |
| Proximal background | 0.1 | 0.1–0.2 |
| Background between sampled radii | 5.8 | 4.8–9.0 |
| Distal background | 0.2 | 0.1–1.0 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 5.6–11.8 | passed |
| 45° | Absent from this structure | 16.8% | 6.1–11.3 | **review** |
| 90° | Absent from this structure | 11.9% | 6.6–9.1 | **review** |
| 22.5° | Possible | 14.8% | 5.6–11.4 | passed |
| 45° | Possible | 23.0% | 6.2–11.2 | passed |
| 90° | Possible | 18.0% | 6.2–8.5 | passed |

Equal weights give a median-map volume of **6.9 km³**. Restricting to the three structures without an extra fan gives **6.9 km³**. Allowing an extra fan in all three gives **6.9 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 194/194 | 1.369 | 1.353 | 1.835 | 84.3% |
| Quadrants | 194/194 | 1.452 | 1.431 | 3.086 | 75.7% |

Geographic sampling warnings occurred in folds 0, 1, 2, 3. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 0, 2, 3. All four folds produced predictions.

Full-data structures requiring numerical review were reference, broad fans. The whole-Earth spatial check passed using 32,768 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.1%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
