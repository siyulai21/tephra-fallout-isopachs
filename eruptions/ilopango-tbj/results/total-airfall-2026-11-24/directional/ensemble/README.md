# Ilopango TBJ · total primary airfall · fixed-direction ensemble

**71.1 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **48.0–456.4 km³**.

The measured triangles contain **1.2 km³** (1.8%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 1,956.5 cm. The largest median thickness on the two display grids is 1,956.5 cm at a source distance of 0.0 km. The evidence panel contains 18 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 2 | 6 | 2.3 | 1.5–13.1 |
| NE | 1 | 10 | 2.0 | 1.3–10.4 |
| E | 11 | 13 | 3.2 | 1.9–31.7 |
| SE | 10 | 11 | 25.4 | 10.2–219.6 |
| S | 4 | 9 | 17.8 | 6.8–172.7 |
| SW | 2 | 6 | 8.2 | 3.9–67.0 |
| W | 24 | 28 | 7.1 | 4.3–32.6 |
| NW | 11 | 20 | 5.1 | 2.8–24.4 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 1.2 | 0.9–1.8 |
| Transition at triangle edges | 1.3 | 1.2–1.9 |
| Proximal background | 6.2 | 3.6–12.5 |
| Background between sampled radii | 23.2 | 18.1–36.4 |
| Distal background | 39.2 | 15.8–416.7 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 15.5% | 48.8–432.4 | passed |
| 45° | Absent from this structure | 16.8% | 49.3–573.3 | passed |
| 90° | Absent from this structure | 11.9% | 49.3–378.2 | passed |
| 22.5° | Possible | 14.8% | 51.4–418.9 | passed |
| 45° | Possible | 23.0% | 48.4–555.8 | passed |
| 90° | Possible | 18.0% | 45.4–270.0 | passed |

Equal weights give a median-map volume of **70.5 km³**. Restricting to the three structures without an extra fan gives **73.4 km³**. Allowing an extra fan in all three gives **69.6 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 168/168 | 1.252 | 1.257 | 1.328 | 93.8% |
| Quadrants | 168/168 | 1.141 | 1.136 | 1.387 | 98.5% |

Geographic sampling warnings occurred in zero folds. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. All four folds produced predictions.

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 4,096 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 1.0%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.00%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
