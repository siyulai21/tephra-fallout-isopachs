# Pinatubo 1991 · climactic Layer C · fixed-direction ensemble

**6.0 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **5.4–7.6 km³**.

The measured triangles contain **0.9 km³** (15.3%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 96.4 cm. The largest median thickness on the two display grids is 96.4 cm at a source distance of 0.0 km. The evidence panel contains 48 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 2 | 0.1 | 0.0–0.2 |
| NE | 12 | 9 | 0.0 | 0.0–0.1 |
| E | 32 | 18 | 0.1 | 0.1–0.1 |
| SE | 22 | 11 | 0.1 | 0.1–0.2 |
| S | 11 | 8 | 0.6 | 0.3–1.3 |
| SW | 35 | 13 | 2.1 | 1.7–3.1 |
| W | 20 | 20 | 2.6 | 2.3–3.1 |
| NW | 14 | 7 | 0.4 | 0.3–0.6 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 0.9 | 0.8–1.1 |
| Transition at triangle edges | 0.2 | 0.2–0.2 |
| Proximal background | 0.1 | 0.1–0.2 |
| Background between sampled radii | 4.6 | 4.1–6.0 |
| Distal background | 0.1 | 0.1–0.4 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 5.3–7.7 | **review** |
| 45° | Absent from this structure | 16.8% | 5.4–7.6 | passed |
| 90° | Absent from this structure | 11.9% | 5.8–7.7 | passed |
| 22.5° | Possible | 14.8% | 5.3–7.6 | passed |
| 45° | Possible | 23.0% | 5.4–7.6 | passed |
| 90° | Possible | 18.0% | 5.7–7.5 | passed |

Equal weights give a median-map volume of **6.0 km³**. Restricting to the three structures without an extra fan gives **5.9 km³**. Allowing an extra fan in all three gives **6.0 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 235/235 | 1.014 | 1.024 | 2.072 | 96.6% |
| Quadrants | 235/235 | 1.511 | 1.513 | 8.789 | 73.5% |

Geographic sampling warnings occurred in folds 0, 1, 2. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 0, 2, 3. All four folds produced predictions.

Full-data structures requiring numerical review were narrow fans. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.2%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.13%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
