# Tambora directional fallout estimate

**112.1 [77.2–172.8] km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **9.9%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. **Numerical warning.** The reference failed at least one sampling or integration check.

![Supported isopach fragments and conditional continuation](isopachs.png)

The left map shades the interpolation footprint and shows 9 measured-bracketed line segments. The right map shows the geometric mean of posterior thickness with dashed contours. Both panels mark exact observations with circles, lower and upper bounds with up and down triangles, and the source with a star. Volume comes from integration of each complete field to the antipode.

The displayed grid reaches 102.0 cm at a source distance of 9.7 km (transition). At the supplied vent it gives 100.5 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Median bulk km³ [95% interval] | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 11.1 [8.6–14.6] | 9.9% |
| Transition around triangle edges | 31.2 [23.8–42.0] | 27.9% |
| Remaining area closer than the nearest exact site | 0.0 [0.0–0.0] | 0.0% |
| Remaining area between the nearest and farthest exact sites | 67.4 [39.0–115.5] | 59.9% |
| Remaining area beyond the farthest exact site | 1.6 [0.3–12.9] | 1.4% |

The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |
|---|---:|---:|---:|---:|---:|
| N | 2 | 0 | 2 | 0 | 15.8 [9.9–26.5] |
| NE | 3 | 0 | 2 | 5,839 | 13.8 [8.5–22.3] |
| E | 4 | 0 | 1 | 9,946 | 12.4 [6.6–21.2] |
| SE | 2 | 0 | 0 | 3,482 | 10.5 [2.4–20.5] |
| S | 1 | 0 | 0 | 2,103 | 11.6 [3.2–21.5] |
| SW | 1 | 0 | 0 | 0 | 12.2 [5.1–21.5] |
| W | 13 | 0 | 8 | 27,954 | 24.9 [16.6–42.4] |
| NW | 0 | 0 | 0 | 0 | 11.3 [4.9–20.9] |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **196.9°** around **171.9°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **116.1 [77.6–193.6] km³**.

## Unmeasured proximal fallout

The nearest accepted site is 15.6 km from the vent. A compact positive ash cap filling a source depression adds a median **0.02 km³** and gives **112.2 [77.4–172.9] km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks carry a warning. Other unobserved proximal deposits can differ. Its matched geographic score is 1.543, with 4 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 105.0 cm, 15.6 km from the vent in retained measurement triangles. With the cap the highest candidate is 1.3 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Median bulk km³ [95% interval] | Numerical checks |
|---|---:|---|
| reference | 112.1 [77.2–172.8] | review |
| heavy tail | 297.1 [171.2–613.5] | pass |
| fixed fourth power | 118.6 [78.7–208.5] | review |
| exponential | 121.3 [79.8–173.6] | pass |
| light tail | 94.8 [72.7–159.1] | review |
| narrow fans | 112.1 [72.7–170.3] | review |
| broad fans | 124.3 [92.7–178.6] | pass |
| possible missing fan | 116.1 [77.6–193.6] | review |
| strict footprint | 112.6 [76.5–173.9] | pass |
| permissive footprint | 118.6 [88.1–171.4] | pass |
| broader background | 121.2 [79.9–175.7] | pass |
| longer transition | 117.6 [83.0–178.1] | pass |
| stronger local departures | 115.5 [79.2–172.1] | pass |

The 13 median estimates span 94.8–297.1 km³ across named assumptions. 8/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.953** (lower is better). Its 95% predictive intervals cover **80.8%** of exact held-out measurements. The quadrant-deletion test scored 26/26 rows with an equal-quadrant score of 1.959. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 0. Quadrant deletion used four compass quadrants. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 3.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
