# Ilopango TBJ · total primary airfall single-structure directional sensitivity

**49.3–573.3 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **1.1%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 17 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 1,861.0 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 1,861.0 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 0.9–1.8 | 1.1% |
| Transition around triangle edges | 1.2–1.9 | 1.3% |
| Remaining area closer than the nearest exact site | 3.7–11.1 | 5.4% |
| Remaining area between the nearest and farthest exact sites | 17.9–34.2 | 21.5% |
| Remaining area beyond the farthest exact site | 20.6–532.9 | 70.3% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 2 | 6 | 1 | 118 | 1.6–7.9 |
| NE | 1 | 10 | 0 | 0 | 1.3–4.0 |
| E | 11 | 13 | 9 | 27 | 1.8–13.9 |
| SE | 10 | 11 | 5 | 577 | 11.1–275.5 |
| S | 4 | 9 | 4 | 4,917 | 6.7–227.6 |
| SW | 2 | 6 | 2 | 1,916 | 3.9–89.3 |
| W | 24 | 28 | 7 | 4,814 | 4.4–39.1 |
| NW | 11 | 20 | 5 | 305 | 2.9–18.5 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **106.7°** around **51.9°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **48.4–555.8 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 5.8 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **49.3–573.3 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.328, with 13 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 1861.0 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 49.3–573.3 | pass |
| heavy tail | 110.8–872.4 | pass |
| fixed fourth power | 54.6–607.7 | pass |
| exponential | 40.0–357.3 | pass |
| light tail | 44.3–478.0 | pass |
| narrow fans | 48.8–432.5 | pass |
| broad fans | 49.5–378.2 | pass |
| possible missing fan | 48.4–555.8 | pass |
| strict footprint | 48.3–540.3 | pass |
| permissive footprint | 49.0–602.1 | pass |
| broader background | 48.7–559.7 | pass |
| longer transition | 49.8–571.5 | pass |
| stronger local departures | 49.9–571.8 | pass |

13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.328** (lower is better). Its 95% predictive intervals cover **83.1%** of exact held-out measurements. The quadrant-deletion test scored 168/168 rows with an equal-quadrant score of 1.387. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 4/4 fitted folds passed sampling checks.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
