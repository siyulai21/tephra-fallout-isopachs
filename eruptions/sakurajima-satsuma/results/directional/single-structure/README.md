# Sakurajima Satsuma (P14) single-structure directional sensitivity

**4.9–62.2 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. These endpoints give the conditional 95% volume interval. A median **1.8%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 23 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 127.0 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 127.0 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 0.2–0.2 | 1.8% |
| Transition around triangle edges | 0.4–0.5 | 4.7% |
| Remaining area closer than the nearest exact site | 0.1–0.4 | 1.6% |
| Remaining area between the nearest and farthest exact sites | 3.1–5.5 | 41.1% |
| Remaining area beyond the farthest exact site | 0.9–56.6 | 50.5% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 2 | 2 | 2 | 25 | 0.3–3.7 |
| NE | 6 | 1 | 3 | 18 | 0.2–2.2 |
| E | 20 | 1 | 6 | 159 | 0.4–5.5 |
| SE | 8 | 0 | 3 | 166 | 0.9–16.0 |
| S | 6 | 2 | 6 | 95 | 1.3–24.9 |
| SW | 3 | 1 | 3 | 151 | 0.4–10.1 |
| W | 3 | 0 | 2 | 4 | 0.4–3.3 |
| NW | 2 | 0 | 0 | 17 | 0.1–2.8 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **97.9°** around **321.2°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives 5.0–62.9 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 9.1 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **4.9–62.2 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.264, with 2 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 127.0 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 4.9–62.2 | pass |
| heavy tail | 26.7–105.9 | pass |
| fixed fourth power | 6.8–29.6 | pass |
| exponential | 4.3–15.3 | pass |
| light tail | 5.1–19.5 | pass |
| narrow fans | 4.0–50.5 | pass |
| broad fans | 6.3–120.1 | pass |
| possible missing fan | 5.0–62.9 | pass |
| strict footprint | 4.7–61.1 | pass |
| permissive footprint | 5.0–62.5 | pass |
| broader background | 5.0–55.3 | pass |
| longer transition | 4.8–61.8 | pass |
| stronger local departures | 4.9–58.1 | pass |

13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.264** (lower is better). Its 95% predictive intervals cover **82.0%** of exact held-out measurements. The quadrant-deletion test scored 57/57 rows with an equal-quadrant score of 1.629. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 4/4 fitted folds passed sampling checks.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
