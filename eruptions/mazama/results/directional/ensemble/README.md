# Mazama · fixed-direction ensemble

**306.8 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **221.5–1,545.2 km³**.

The measured triangles contain **20.5 km³** (6.7%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 6,465.5 cm. The largest median thickness on the two display grids is 6,465.5 cm at a source distance of 0.0 km. The evidence panel contains 22 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 20 | 91 | 32.6 | 22.6–57.3 |
| NE | 12 | 99 | 125.9 | 63.5–1,215.8 |
| E | 10 | 38 | 32.1 | 24.2–54.1 |
| SE | 3 | 22 | 30.2 | 20.5–58.9 |
| S | 0 | 1 | 21.7 | 12.1–68.6 |
| SW | 0 | 0 | 21.8 | 12.3–236.9 |
| W | 0 | 1 | 20.8 | 11.5–55.5 |
| NW | 1 | 2 | 21.7 | 12.2–48.0 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 20.5 | 16.9–26.9 |
| Transition at triangle edges | 56.5 | 43.6–91.4 |
| Proximal background | 79.8 | 38.2–199.8 |
| Background between sampled radii | 93.1 | 73.8–194.5 |
| Distal background | 56.9 | 8.0–1,247.2 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 16.1% | 232.6–613.4 | passed |
| 45° | Absent from this structure | 15.7% | 246.7–2,472.5 | passed |
| 90° | Absent from this structure | 12.3% | 199.0–499.8 | passed |
| 22.5° | Possible | 15.7% | 246.7–849.4 | **review** |
| 45° | Possible | 21.2% | 263.0–2,578.0 | passed |
| 90° | Possible | 18.9% | 214.5–672.0 | passed |

Equal weights give a median-map volume of **302.2 km³**. Restricting to the three structures without an extra fan gives **304.3 km³**. Allowing an extra fan in all three gives **316.1 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 300/300 | 0.781 | 0.784 | 1.002 | 87.0% |
| Quadrants | 300/300 | 1.053 | 1.057 | 1.497 | 91.3% |

Geographic sampling warnings occurred in folds 0, 2. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 1. Unscored folds were 2 (empty).

Full-data structures requiring numerical review were narrow missing fan. The whole-Earth spatial check passed using 16,384 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.2%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.00%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
