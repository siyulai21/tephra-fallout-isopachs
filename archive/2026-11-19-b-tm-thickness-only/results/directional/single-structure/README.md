# B-Tm single-structure directional sensitivity

**55.5 [33.7–134.7] km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **8.3%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 3 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 118.5 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 118.5 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Median bulk km³ [95% interval] | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 4.6 [3.6–6.0] | 8.3% |
| Transition around triangle edges | 3.8 [3.0–5.0] | 6.9% |
| Remaining area closer than the nearest exact site | 0.6 [0.2–1.5] | 1.0% |
| Remaining area between the nearest and farthest exact sites | 32.5 [19.2–94.2] | 63.4% |
| Remaining area beyond the farthest exact site | 9.0 [2.8–61.1] | 16.8% |

The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |
|---|---:|---:|---:|---:|---:|
| N | 0 | 1 | 0 | 0 | 2.0 [0.2–10.6] |
| NE | 0 | 2 | 0 | 0 | 3.6 [1.6–12.5] |
| E | 64 | 0 | 32 | 96,131 | 34.5 [26.2–72.5] |
| SE | 0 | 1 | 0 | 0 | 4.9 [2.0–14.9] |
| S | 0 | 0 | 0 | 0 | 2.0 [0.2–10.6] |
| SW | 0 | 0 | 0 | 0 | 2.0 [0.2–10.6] |
| W | 0 | 0 | 0 | 0 | 2.0 [0.2–10.6] |
| NW | 0 | 0 | 0 | 0 | 2.0 [0.2–10.6] |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **341.0°** around **269.7°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **100.8 [40.4–1,545.0] km³**.

## Unmeasured proximal fallout

The nearest accepted site is 2.3 km from the vent. A compact positive ash cap filling a source depression adds a median **0.00 km³** and gives **55.5 [33.7–134.7] km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.425, with 17 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 118.5 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Median bulk km³ [95% interval] | Numerical checks |
|---|---:|---|
| reference | 55.5 [33.7–134.7] | pass |
| heavy tail | 118.9 [76.4–276.3] | pass |
| fixed fourth power | 61.7 [41.4–133.9] | pass |
| exponential | 48.1 [32.0–122.3] | pass |
| light tail | 53.2 [35.5–125.3] | pass |
| narrow fans | 50.5 [26.8–126.7] | pass |
| broad fans | 60.2 [44.8–120.0] | pass |
| possible missing fan | 100.8 [40.4–1,545.0] | pass |
| strict footprint | 45.3 [32.5–99.2] | pass |
| permissive footprint | 54.1 [33.8–134.5] | pass |
| broader background | 60.6 [34.7–138.7] | pass |
| longer transition | 54.3 [33.5–131.1] | pass |
| stronger local departures | 55.9 [34.6–137.6] | pass |

The 13 median estimates span 45.3–118.9 km³ across named assumptions. 13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.425** (lower is better). Its 95% predictive intervals cover **93.8%** of exact held-out measurements. The quadrant-deletion test scored 68/68 rows with an equal-quadrant score of 1.448. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 2/2 fitted folds passed sampling checks. Unscored folds were 2 (empty), 3 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
