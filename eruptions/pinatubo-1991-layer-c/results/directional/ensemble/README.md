# Pinatubo 1991 (Layer C) · fixed-direction ensemble

**6.7 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **5.9–10.9 km³**.

The measured triangles contain **0.6 km³** (8.4%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 98.3 cm. The largest median thickness on the two display grids is 98.3 cm at a source distance of 0.0 km. The evidence panel contains 44 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 2 | 0.0 | 0.0–0.2 |
| NE | 12 | 2 | 0.0 | 0.0–0.1 |
| E | 32 | 14 | 0.1 | 0.1–0.1 |
| SE | 22 | 9 | 0.1 | 0.1–0.1 |
| S | 11 | 8 | 0.6 | 0.3–1.7 |
| SW | 35 | 8 | 2.3 | 1.6–5.3 |
| W | 24 | 12 | 3.4 | 2.9–4.1 |
| NW | 13 | 3 | 0.2 | 0.1–1.1 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 0.6 | 0.5–0.6 |
| Transition at triangle edges | 0.2 | 0.2–0.2 |
| Proximal background | 0.1 | 0.1–0.2 |
| Background between sampled radii | 5.7 | 4.9–9.0 |
| Distal background | 0.2 | 0.1–1.0 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 5.6–12.3 | passed |
| 45° | Absent from this structure | 16.8% | 6.1–11.5 | **review** |
| 90° | Absent from this structure | 11.9% | 6.6–8.3 | passed |
| 22.5° | Possible | 14.8% | 5.7–11.6 | passed |
| 45° | Possible | 23.0% | 6.2–11.3 | **review** |
| 90° | Possible | 18.0% | 6.1–7.8 | passed |

Equal weights give a median-map volume of **6.7 km³**. Restricting to the three structures without an extra fan gives **6.8 km³**. Allowing an extra fan in all three gives **6.7 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 208/208 | 1.462 | 1.465 | 2.402 | 85.3% |
| Quadrants | 208/208 | 1.437 | 1.416 | 3.220 | 72.0% |

Geographic sampling warnings occurred in folds 1, 2. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 0, 2, 3. All four folds produced predictions.

Full-data structures requiring numerical review were reference, possible missing fan. The whole-Earth spatial check passed using 32,768 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.3%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
