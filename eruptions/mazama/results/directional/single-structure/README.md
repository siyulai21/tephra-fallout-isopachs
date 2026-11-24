# Mazama single-structure directional sensitivity

**246.7–2,472.8 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **5.0%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 31 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 6,413.8 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 6,413.8 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 17.4–26.5 | 5.0% |
| Transition around triangle edges | 42.9–83.5 | 14.0% |
| Remaining area closer than the nearest exact site | 37.8–180.0 | 18.3% |
| Remaining area between the nearest and farthest exact sites | 73.7–129.1 | 23.2% |
| Remaining area beyond the farthest exact site | 29.6–2,182.8 | 37.5% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 20 | 91 | 9 | 57,235 | 24.2–53.1 |
| NE | 12 | 99 | 6 | 108,770 | 92.1–2,268.3 |
| E | 10 | 38 | 8 | 3,459 | 23.2–49.9 |
| SE | 3 | 22 | 0 | 43,016 | 20.5–53.0 |
| S | 0 | 1 | 0 | 0 | 11.7–41.3 |
| SW | 0 | 0 | 0 | 0 | 10.9–40.5 |
| W | 0 | 1 | 0 | 0 | 10.9–40.5 |
| NW | 1 | 2 | 0 | 3,474 | 11.9–41.7 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **240.8°** around **221.5°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **263.0–2,578.2 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 3.3 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **246.7–2,472.8 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.001, with 26 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 6413.8 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 246.7–2,472.8 | pass |
| heavy tail | 400.8–2,590.5 | pass |
| fixed fourth power | 265.8–2,113.5 | pass |
| exponential | 244.1–2,365.0 | pass |
| light tail | 249.2–2,061.7 | pass |
| narrow fans | 232.6–612.6 | pass |
| broad fans | 199.1–499.8 | pass |
| possible missing fan | 263.0–2,578.2 | pass |
| strict footprint | 246.0–1,578.4 | pass |
| permissive footprint | 223.2–1,100.0 | pass |
| broader background | 244.1–2,387.1 | pass |
| longer transition | 240.5–4,125.6 | pass |
| stronger local departures | 231.2–1,763.6 | pass |

13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.002** (lower is better). Its 95% predictive intervals cover **87.0%** of exact held-out measurements. The quadrant-deletion test scored 300/300 rows with an equal-quadrant score of 1.497. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 2. Quadrant deletion used four compass quadrants. 3/3 fitted folds passed sampling checks. Unscored folds were 2 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
