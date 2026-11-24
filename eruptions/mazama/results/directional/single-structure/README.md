# Mazama single-structure directional sensitivity

**269.3–635.3 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **8.8%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 24 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 6,636.3 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 6,636.3 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 27.1–43.8 | 8.8% |
| Transition around triangle edges | 55.4–118.5 | 20.3% |
| Remaining area closer than the nearest exact site | 43.2–224.6 | 23.8% |
| Remaining area between the nearest and farthest exact sites | 68.8–132.9 | 24.0% |
| Remaining area beyond the farthest exact site | 31.3–248.2 | 20.8% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 31 | 56 | 15 | 69,280 | 49.7–130.0 |
| NE | 10 | 73 | 4 | 69,510 | 82.8–236.5 |
| E | 13 | 30 | 10 | 33,313 | 29.1–66.8 |
| SE | 3 | 22 | 0 | 67,733 | 22.6–59.9 |
| S | 0 | 1 | 0 | 0 | 14.4–52.5 |
| SW | 0 | 0 | 0 | 0 | 13.7–51.9 |
| W | 0 | 1 | 0 | 0 | 13.8–51.9 |
| NW | 1 | 3 | 0 | 3,474 | 15.5–55.4 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **240.8°** around **221.5°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **285.6–1,051.2 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 3.3 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **269.3–635.3 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 0.963, with 27 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 6636.3 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 269.3–635.3 | pass |
| heavy tail | 466.6–1,124.5 | pass |
| fixed fourth power | 296.9–701.7 | pass |
| exponential | 267.2–624.6 | pass |
| light tail | 277.7–647.8 | pass |
| narrow fans | 330.1–1,257.7 | review |
| broad fans | 238.0–582.5 | pass |
| possible missing fan | 285.6–1,051.2 | pass |
| strict footprint | 317.8–841.4 | pass |
| permissive footprint | 316.2–973.1 | pass |
| broader background | 269.5–621.3 | pass |
| longer transition | 242.4–531.8 | pass |
| stronger local departures | 260.0–641.1 | pass |

12/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **0.963** (lower is better). Its 95% predictive intervals cover **81.0%** of exact held-out measurements. The quadrant-deletion test scored 244/244 rows with an equal-quadrant score of 1.177. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 3. Quadrant deletion used four compass quadrants. 3/3 fitted folds passed sampling checks. Unscored folds were 2 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
