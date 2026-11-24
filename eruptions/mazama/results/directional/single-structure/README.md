# Mazama single-structure directional sensitivity

**155.9–429.3 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **16.3%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 37 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 3,802.6 cm at a source distance of 15.1 km (background continuation). At the supplied vent it gives 2,157.6 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 24.3–74.0 | 16.3% |
| Transition around triangle edges | 50.3–148.9 | 33.9% |
| Remaining area closer than the nearest exact site | 13.1–118.1 | 16.9% |
| Remaining area between the nearest and farthest exact sites | 39.8–115.5 | 27.0% |
| Remaining area beyond the farthest exact site | 3.1–39.6 | 3.9% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 13 | 89 | 6 | 36,440 | 15.4–32.2 |
| NE | 3 | 89 | 1 | 1,662 | 27.2–67.0 |
| E | 11 | 36 | 9 | 4,377 | 34.0–92.3 |
| SE | 3 | 22 | 0 | 44,251 | 18.0–60.1 |
| S | 0 | 1 | 0 | 0 | 11.9–68.1 |
| SW | 0 | 0 | 0 | 0 | 11.6–70.1 |
| W | 0 | 1 | 0 | 0 | 11.1–64.4 |
| NW | 1 | 2 | 0 | 6,907 | 6.6–23.7 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **240.8°** around **221.5°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **164.2–673.9 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 3.3 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **156.2–430.1 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 0.645, with 26 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 3802.6 cm, 15.1 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 155.9–429.3 | pass |
| heavy tail | 204.7–392.8 | pass |
| fixed fourth power | 152.1–363.2 | pass |
| exponential | 154.5–449.6 | pass |
| light tail | 148.4–376.3 | pass |
| narrow fans | 201.8–590.8 | pass |
| broad fans | 153.2–420.4 | pass |
| possible missing fan | 164.2–673.9 | review |
| strict footprint | 166.7–491.1 | pass |
| permissive footprint | 153.8–410.0 | pass |
| broader background | 155.0–428.0 | review |
| longer transition | 146.9–363.6 | pass |
| stronger local departures | 128.7–357.7 | pass |

11/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **0.670** (lower is better). Its 95% predictive intervals cover **83.9%** of exact held-out measurements. The quadrant-deletion test scored 271/271 rows with an equal-quadrant score of 0.632. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 2/3 fitted folds passed sampling checks, with a numerical warning in fold 1. Unscored folds were 2 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
