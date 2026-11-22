# Pinatubo 1991 (Layer C) single-structure directional sensitivity

**5.9–10.1 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **7.4%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 58 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 89.1 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 89.1 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 0.5–0.6 | 7.4% |
| Transition around triangle edges | 0.2–0.2 | 2.4% |
| Remaining area closer than the nearest exact site | 0.1–0.1 | 1.5% |
| Remaining area between the nearest and farthest exact sites | 5.0–8.6 | 86.0% |
| Remaining area beyond the farthest exact site | 0.1–0.6 | 2.6% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 1 | 2 | 0 | 0 | 0.0–0.1 |
| NE | 12 | 2 | 4 | 15 | 0.0–0.1 |
| E | 30 | 9 | 11 | 206 | 0.1–0.1 |
| SE | 19 | 10 | 12 | 236 | 0.1–0.1 |
| S | 10 | 7 | 6 | 62 | 0.4–1.8 |
| SW | 32 | 8 | 14 | 4,344 | 1.7–4.4 |
| W | 24 | 12 | 18 | 18,450 | 2.9–4.3 |
| NW | 11 | 4 | 5 | 23 | 0.1–0.3 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **102.6°** around **350.2°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **5.9–10.2 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 8.7 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **5.9–10.1 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.907, with 3 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 89.1 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 5.9–10.1 | pass |
| heavy tail | 17.0–39.4 | pass |
| fixed fourth power | 7.0–12.5 | pass |
| exponential | 5.9–10.1 | pass |
| light tail | 6.1–10.0 | pass |
| narrow fans | 5.9–13.1 | review |
| broad fans | 6.7–9.1 | pass |
| possible missing fan | 5.9–10.2 | review |
| strict footprint | 5.9–10.1 | review |
| permissive footprint | 5.7–8.5 | pass |
| broader background | 5.9–9.9 | pass |
| longer transition | 5.9–10.1 | pass |
| stronger local departures | 5.9–10.2 | pass |

10/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.907** (lower is better). Its 95% predictive intervals cover **79.9%** of exact held-out measurements. The quadrant-deletion test scored 193/193 rows with an equal-quadrant score of 3.125. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 2.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
