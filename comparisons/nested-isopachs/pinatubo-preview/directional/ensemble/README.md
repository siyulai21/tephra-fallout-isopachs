# Pinatubo 1991 Layer C · fixed-direction ensemble

**5.2 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **4.3–8.5 km³**.

The measured triangles contain **0.5 km³** (10.2%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 105.3 cm. The largest median thickness on the two display grids is 105.3 cm at a source distance of 0.0 km. The evidence panel contains 78 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 2 | 0.1 | 0.0–0.2 |
| NE | 8 | 2 | 0.1 | 0.0–0.1 |
| E | 28 | 8 | 0.1 | 0.1–0.1 |
| SE | 17 | 7 | 0.1 | 0.1–0.2 |
| S | 7 | 6 | 0.5 | 0.3–1.9 |
| SW | 20 | 4 | 1.8 | 1.2–3.3 |
| W | 14 | 4 | 2.5 | 2.1–3.8 |
| NW | 7 | 2 | 0.1 | 0.1–1.2 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 0.5 | 0.4–0.6 |
| Transition at triangle edges | 0.2 | 0.2–0.2 |
| Proximal background | 0.1 | 0.1–0.2 |
| Background between sampled radii | 4.3 | 3.4–7.2 |
| Distal background | 0.1 | 0.0–0.6 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 4.1–7.7 | **review** |
| 45° | Absent from this structure | 16.8% | 4.6–7.6 | **review** |
| 90° | Absent from this structure | 11.9% | 6.2–9.1 | passed |
| 22.5° | Possible | 14.8% | 4.2–8.0 | **review** |
| 45° | Possible | 23.0% | 4.6–7.6 | passed |
| 90° | Possible | 18.0% | 5.8–8.5 | **review** |

Equal weights give a median-map volume of **5.2 km³**. Restricting to the three structures without an extra fan gives **5.1 km³**. Allowing an extra fan in all three gives **5.2 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 137/137 | 2.473 | 2.476 | 14.917 | 84.3% |
| Quadrants | 137/137 | 1.730 | 1.718 | 3.493 | 66.7% |

Geographic sampling warnings occurred in folds 1, 2. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 0, 1, 2, 3. All four folds produced predictions.

Full-data structures requiring numerical review were narrow fans, reference, narrow missing fan, broad missing fan. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.7%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.00%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
