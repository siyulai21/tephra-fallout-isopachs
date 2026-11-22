# B-Tm single-structure directional sensitivity

**29.5–49.7 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. These endpoints give the conditional 95% volume interval. A median **13.4%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 4 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 198.6 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 198.6 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 3.9–6.5 | 13.4% |
| Transition around triangle edges | 3.0–4.8 | 10.2% |
| Remaining area closer than the nearest exact site | 0.1–1.7 | 0.9% |
| Remaining area between the nearest and farthest exact sites | 17.5–31.6 | 63.2% |
| Remaining area beyond the farthest exact site | 1.9–15.4 | 10.9% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 0 | 1 | 0 | 0 | 0.0–0.3 |
| NE | 0 | 3 | 0 | 0 | 1.2–2.1 |
| E | 64 | 0 | 32 | 96,131 | 26.2–44.1 |
| SE | 0 | 1 | 0 | 0 | 1.4–3.5 |
| S | 0 | 0 | 0 | 0 | 0.0–0.3 |
| SW | 0 | 0 | 0 | 0 | 0.0–0.3 |
| W | 0 | 1 | 0 | 0 | 0.0–0.3 |
| NW | 0 | 1 | 0 | 0 | 0.0–0.3 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **341.0°** around **269.7°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives 28.7–46.3 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 2.3 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **29.5–49.7 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.274, with 17 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 198.6 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 29.5–49.7 | pass |
| heavy tail | 62.0–103.7 | pass |
| fixed fourth power | 35.9–53.6 | pass |
| exponential | 29.0–46.4 | pass |
| light tail | 31.4–47.5 | pass |
| narrow fans | 22.3–37.8 | pass |
| broad fans | 41.7–69.8 | pass |
| possible missing fan | 28.7–46.3 | pass |
| strict footprint | 29.9–48.0 | pass |
| permissive footprint | 29.7–48.6 | pass |
| broader background | 29.5–48.0 | pass |
| longer transition | 30.4–50.6 | pass |
| stronger local departures | 30.1–48.6 | pass |

13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.274** (lower is better). Its 95% predictive intervals cover **93.8%** of exact held-out measurements. The quadrant-deletion test scored 71/71 rows with an equal-quadrant score of 4.508. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 3/3 fitted folds passed sampling checks. Unscored folds were 2 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
