# Mazama · fixed-direction ensemble

**356.4 km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. The conditional 95% volume interval is **267.4–1,079.0 km³**.

The measured triangles contain **32.6 km³** (9.2%) of the median map. The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.

![Measured-bracketed median fragments and expanded median isopachs](isopachs.png)

The star marks the supplied vent. Its median thickness is 6,528.3 cm. The largest median thickness on the two display grids is 6,528.3 cm at a source distance of 0.0 km. The evidence panel contains 20 measured-bracketed triangle and level pairs. The right panel includes conditional continuation.

![Local median thickness and pointwise 95% uncertainty](local.png)

![Median thickness over the Earth](world.png)

## Direction and observational support

| Direction | Exact rows | Bounds | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|
| N | 31 | 56 | 67.9 | 50.1–238.2 |
| NE | 10 | 73 | 109.2 | 64.3–505.5 |
| E | 13 | 30 | 40.3 | 29.6–82.1 |
| SE | 3 | 22 | 32.9 | 22.3–64.7 |
| S | 0 | 1 | 26.1 | 14.5–75.0 |
| SW | 0 | 0 | 26.9 | 14.9–260.3 |
| W | 0 | 1 | 25.5 | 14.0–66.8 |
| NW | 1 | 3 | 27.6 | 16.5–76.6 |

| Domain | Median-map volume (km³) | Conditional 95% volume interval (km³) |
|---|---:|---:|
| Measured triangles | 32.6 | 27.0–45.3 |
| Transition at triangle edges | 75.8 | 55.8–125.6 |
| Proximal background | 91.1 | 41.6–230.5 |
| Background between sampled radii | 93.2 | 72.9–197.4 |
| Distal background | 63.7 | 15.1–723.0 |

The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.

## Assumptions and weights

| Fan half-width | Additional fan in largest distal gap | Predictive weight | Conditional 95% volume interval (km³) | Fit and mass checks |
|---:|---|---:|---:|---|
| 22.5° | Absent from this structure | 16.1% | 330.1–1,258.4 | **review** |
| 45° | Absent from this structure | 15.7% | 269.3–636.5 | passed |
| 90° | Absent from this structure | 12.3% | 238.0–582.5 | passed |
| 22.5° | Possible | 15.7% | 347.7–1,457.6 | passed |
| 45° | Possible | 21.2% | 285.6–1,053.0 | passed |
| 90° | Possible | 18.9% | 258.4–828.9 | passed |

Equal weights give a median-map volume of **354.5 km³**. Restricting to the three structures without an extra fan gives **356.5 km³**. Allowing an extra fan in all three gives **365.4 km³**. These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.

Independent calibration cases with usable folds were tambora, aso-4, b-tm, ytt, minoan. Weights describe held-out thickness prediction. They depend on the small eruption catalogue.

## Validation and numerical checks

| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|
| Geographic | 244/244 | 0.844 | 0.849 | 0.963 | 86.2% |
| Quadrants | 244/244 | 0.898 | 0.896 | 1.177 | 96.6% |

Geographic sampling warnings occurred in folds 1, 3. All four folds produced predictions.

Quadrants sampling warnings occurred in folds 1. Unscored folds were 2 (empty).

Full-data structures requiring numerical review were narrow fans. The whole-Earth spatial check passed using 16,384 points in each of two scrambles. The interleaved posterior-draw check passed (largest change 0.7%). The resolved isopach-area split checks passed. The independent same-fields mean-integral discrepancy was 0.02%.

The shared complete-contour audit found **0 qualifying policies out of 3**. The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.

[Numerical report, fold identities and weight sensitivity](report.json) · [Complete contour decisions](complete-contour-audit.json) · [All fragment decisions](fragment-decisions.csv) · [Mapped fragment vertices](supported-fragments.csv) · [Held-out predictions](heldout.csv)

[Reconstruct the median field](map-fields.npz) · [Map grids and pointwise quantiles](median-grids.npz) · [Joint volume by chain, support class and compass sector](coherent-mass.npz) · [Checksums](artifacts.json)
