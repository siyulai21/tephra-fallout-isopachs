# Aso-4 directional fallout

**659.1 km³ of bulk fallout** is the whole-Earth integral of the principal median isopach map. The uncertain-fan-direction ensemble uses 87 quantitative observations and bounds. Integrating complete simulated deposits gives a median of **916.3 km³** and a conditional 95% interval of **297.9–3,511.5 km³**.

![Principal median isopachs with uncertain fan direction](azimuth/isopachs.png)

Measured triangles account for **3.8%** of the map volume. Solid contour fragments cross triangles where observations bracket the thickness. Dashes trace the continuation through gaps. A poorly sampled direction can still contain an unobserved fan.

## Predictive and numerical checks

| Held-out observations | Fixed-direction control | Uncertain fan direction | Exact-site 95% coverage |
|---|---:|---:|---:|
| Geographic regions | 1.500 | 1.502 | 75/83 (90.4%) |
| Source quadrants | 1.302 | 1.566 | 78/83 (94.0%) |

Lower scores indicate better predictions. The table includes every estimable fold. Passing-only scores, unscored folds and site-level intervals are available in the detailed report. Across the six benchmark eruptions, uncertain directions slightly improved geographic scores and worsened whole-quadrant scores.

[Sector areas, directional reach, support audit and alternative assumptions](azimuth/README.md) · [Principal numerical result](azimuth/report.json) · [Whole-Earth map](azimuth/world.png) · [Held-out site predictions](azimuth/heldout.csv)

## Directional sensitivities

Fixing fan directions under the same weights gives a median-map integral of 779.7 km³. The earlier fixed-direction ensemble has its own geographic calibration. [Earlier six-structure reconstruction](ensemble/README.md) · [Single-structure and proximal-cap study](single-structure.md)
