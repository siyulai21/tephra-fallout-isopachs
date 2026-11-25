# B-Tm single-structure directional sensitivity

**27.1–45.8 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **14.8%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 9 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 56.9 cm at a source distance of 32.7 km (transition). At the supplied vent it gives 55.7 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 3.9–6.3 | 14.8% |
| Transition around triangle edges | 3.0–4.6 | 11.3% |
| Remaining area closer than the nearest exact site | 0.0–0.0 | 0.0% |
| Remaining area between the nearest and farthest exact sites | 16.1–26.9 | 62.6% |
| Remaining area beyond the farthest exact site | 1.6–16.3 | 10.3% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 1 | 0 | 0 | 0 | 0.0–0.1 |
| NE | 2 | 1 | 0 | 162 | 1.1–2.8 |
| E | 64 | 0 | 34 | 96,108 | 24.2–40.4 |
| SE | 0 | 1 | 0 | 0 | 1.3–3.4 |
| S | 0 | 0 | 0 | 0 | 0.0–0.1 |
| SW | 0 | 0 | 0 | 0 | 0.0–0.1 |
| W | 0 | 1 | 0 | 0 | 0.0–0.1 |
| NW | 0 | 1 | 0 | 0 | 0.0–0.1 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **340.9°** around **269.7°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **26.5–40.7 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 2.3 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **27.1–45.8 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.314, with 17 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 59.0 cm, 30.3 km from the vent in retained measurement triangles. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 27.1–45.8 | pass |
| heavy tail | 60.5–102.7 | pass |
| fixed fourth power | 33.1–49.5 | pass |
| exponential | 26.4–39.8 | pass |
| light tail | 28.5–41.7 | pass |
| narrow fans | 22.3–44.3 | pass |
| broad fans | 33.8–55.7 | pass |
| possible missing fan | 26.5–40.7 | pass |
| strict footprint | 25.7–41.1 | pass |
| permissive footprint | 27.3–45.6 | pass |
| broader background | 26.9–44.4 | pass |
| longer transition | 27.0–47.9 | pass |
| stronger local departures | 27.9–47.6 | pass |

13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.314** (lower is better). Its 95% predictive intervals cover **94.0%** of exact held-out measurements. The quadrant-deletion test scored 71/71 rows with an equal-quadrant score of 4.341. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 3/3 fitted folds passed sampling checks. Unscored folds were 2 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
