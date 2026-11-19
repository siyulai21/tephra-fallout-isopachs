# YTT directional fallout

**900.6 km³ of bulk fallout** is the whole-Earth integral of the principal median isopach map. The uncertain-fan-direction ensemble uses 46 quantitative observations and bounds. Integrating complete simulated deposits gives a median of **1,272.1 km³** and a conditional 95% interval of **736.5–3,660.9 km³**.

![Principal median isopachs with uncertain fan direction](azimuth/isopachs.png)

Measured triangles account for **21.7%** of the map volume. Solid contour fragments cross triangles where observations bracket the thickness. Dashes trace the continuation through gaps. A poorly sampled direction can still contain an unobserved fan.

## Predictive and numerical checks

| Held-out observations | Fixed-direction control | Uncertain fan direction | Exact-site 95% coverage |
|---|---:|---:|---:|
| Geographic regions | 1.781 | 1.531 | 46/46 (100.0%) |
| Source quadrants | 3.056 | 2.969 | 19/46 (41.3%) |

Lower scores indicate better predictions. The table includes every estimable fold. Passing-only scores, unscored folds and site-level intervals are available in the detailed report. Across the six benchmark eruptions, uncertain directions slightly improved geographic scores and worsened whole-quadrant scores.

[Sector areas, directional reach, support audit and alternative assumptions](azimuth/README.md) · [Principal numerical result](azimuth/report.json) · [Whole-Earth map](azimuth/world.png) · [Held-out site predictions](azimuth/heldout.csv)

## Directional sensitivities

Fixing fan directions under the same weights gives a median-map integral of 1,005.6 km³. The earlier fixed-direction ensemble has its own geographic calibration. [Earlier six-structure reconstruction](ensemble/README.md) · [Single-structure and proximal-cap study](single-structure.md)
