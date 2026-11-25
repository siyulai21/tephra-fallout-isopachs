# Minoan · fixed-direction ensemble

**35.2 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **25.9–150.7 km³**.

The measured triangles contain **2.7 km³** (7.7%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 31.6 cm. The largest median thickness on the two display grids is 48.4 cm at a source distance of 159.5 km. The evidence panel contains 38 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 1 | 0 | 1.5 | 0.5–15.1 |
| NE | 18 | 0 | 4.7 | 3.1–10.1 |
| E | 38 | 0 | 16.6 | 8.1–92.6 |
| SE | 25 | 0 | 7.7 | 4.7–29.1 |
| S | 8 | 0 | 1.8 | 0.8–13.1 |
| SW | 4 | 0 | 1.1 | 0.4–6.3 |
| W | 4 | 0 | 0.9 | 0.4–5.2 |
| NW | 3 | 0 | 0.9 | 0.3–6.3 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 2.7 | 2.4–3.6 |
| Transition at triangle edges | 3.9 | 3.4–4.9 |
| Proximal background | 0.0 | 0.0–0.1 |
| Background between sampled radii | 24.5 | 18.2–52.7 |
| Distal background | 4.1 | 0.6–95.5 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 17.2% | 29.0–184.3 | passed |
| 45° | Absent from this structure | 17.0% | 26.8–147.2 | passed |
| 90° | Absent from this structure | 11.0% | 24.2–62.0 | passed |
| 22.5° | Possible | 16.5% | 30.3–186.3 | passed |
| 45° | Possible | 23.0% | 26.6–149.4 | passed |
| 90° | Possible | 15.3% | 24.5–57.8 | passed |

Equal weights give a median-map volume of **34.0 km³**. Restricting to the three structures without an extra fan gives **34.9 km³**. Allowing an extra fan in all three gives **35.4 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were mazama, tambora, aso-4, b-tm, ytt. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 101/101 | 1.820 | 1.806 | 1.857 | 91.1% |
| Quadrants | 101/101 | 2.151 | 2.126 | 2.739 | 80.2% |

Geographic sampling warnings occurred in zero folds. All four folds produced predictions.

Quadrants sampling warnings occurred in zero folds. All four folds produced predictions.

All six full-data structures passed their numerical checks. The whole-Earth spatial check passed using 8,192 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.1%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.00%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
