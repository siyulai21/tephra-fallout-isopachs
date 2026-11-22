# Minoan · fixed-direction ensemble

**33.7 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **26.6–129.1 km³**.

The measured triangles contain **2.4 km³** (7.3%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 55.1 cm. The largest median thickness on the two display grids is 55.1 cm at a source distance of 0.0 km. The evidence panel contains 30 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 0 | 1.7 | 0.6–17.5 |
| NE | 18 | 0 | 4.9 | 3.3–9.7 |
| E | 34 | 4 | 14.8 | 7.8–78.4 |
| SE | 25 | 0 | 7.2 | 4.6–24.9 |
| S | 8 | 0 | 1.8 | 0.9–9.9 |
| SW | 3 | 1 | 1.3 | 0.6–7.0 |
| W | 2 | 2 | 1.1 | 0.5–7.0 |
| NW | 1 | 2 | 0.8 | 0.2–4.9 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 2.4 | 2.2–3.4 |
| Transition at triangle edges | 3.8 | 3.3–4.8 |
| Proximal background | 0.8 | 0.5–2.3 |
| Background between sampled radii | 23.8 | 18.4–51.1 |
| Distal background | 2.9 | 0.5–74.5 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 17.2% | 28.3–151.1 | passed |
| 45° | Absent from this structure | 17.0% | 26.2–126.9 | passed |
| 90° | Absent from this structure | 11.0% | 25.7–67.6 | passed |
| 22.5° | Possible | 16.5% | 28.9–163.5 | passed |
| 45° | Possible | 23.0% | 26.6–127.7 | passed |
| 90° | Possible | 15.3% | 26.0–61.9 | passed |

Equal weights give a median-map volume of **32.9 km³**. Restricting to the three structures without an extra fan gives **33.2 km³**. Allowing an extra fan in all three gives **34.1 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 101/101 | 1.695 | 1.684 | 1.732 | 91.3% |
| Quadrants | 101/101 | 2.156 | 2.128 | 2.502 | 69.6% |

Geographic sampling warnings occurred in zero folds. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. All four folds produced predictions.

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.1%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.01%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
