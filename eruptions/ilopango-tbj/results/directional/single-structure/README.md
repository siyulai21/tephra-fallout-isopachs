# Ilopango TBJ (Unit G) single-structure directional sensitivity

**69.9 [35.9–449.3] km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **2.3%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 19 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 439.6 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 439.6 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Median bulk km³ [95% interval] | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 1.6 [1.3–1.9] | 2.3% |
| Transition around triangle edges | 3.2 [2.7–3.8] | 4.6% |
| Remaining area closer than the nearest exact site | 0.5 [0.2–1.3] | 0.7% |
| Remaining area between the nearest and farthest exact sites | 26.4 [20.0–36.2] | 37.9% |
| Remaining area beyond the farthest exact site | 37.5 [7.1–413.6] | 54.3% |

The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |
|---|---:|---:|---:|---:|---:|
| N | 3 | 0 | 2 | 114 | 1.6 [0.9–23.0] |
| NE | 2 | 1 | 0 | 151 | 2.6 [1.2–22.7] |
| E | 9 | 1 | 7 | 165 | 7.8 [2.2–52.3] |
| SE | 5 | 1 | 4 | 644 | 20.3 [6.0–191.9] |
| S | 12 | 0 | 5 | 5,601 | 13.0 [6.0–89.8] |
| SW | 2 | 0 | 2 | 4,013 | 7.9 [3.7–51.8] |
| W | 12 | 1 | 3 | 4,685 | 5.7 [3.5–34.9] |
| NW | 7 | 0 | 3 | 600 | 4.4 [2.4–32.9] |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **96.5°** around **57.0°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **63.1 [36.5–239.4] km³**.

## Unmeasured proximal fallout

The nearest accepted site is 10.8 km from the vent. A compact positive ash cap filling a source depression adds a median **0.00 km³** and gives **69.9 [35.9–449.3] km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.728, with 8 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 439.6 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Median bulk km³ [95% interval] | Numerical checks |
|---|---:|---|
| reference | 69.9 [35.9–449.3] | pass |
| heavy tail | 197.2 [115.5–551.5] | pass |
| fixed fourth power | 72.4 [44.0–233.6] | pass |
| exponential | 47.1 [32.1–143.3] | pass |
| light tail | 53.9 [35.7–146.5] | pass |
| narrow fans | 56.2 [29.9–341.5] | pass |
| broad fans | 89.0 [40.2–525.3] | pass |
| possible missing fan | 63.1 [36.5–239.4] | pass |
| strict footprint | 53.0 [33.0–191.1] | pass |
| permissive footprint | 72.2 [35.0–478.6] | pass |
| broader background | 68.8 [35.4–390.7] | pass |
| longer transition | 68.8 [35.8–443.2] | pass |
| stronger local departures | 71.7 [35.4–493.7] | pass |

The 13 median estimates span 47.1–197.2 km³ across named assumptions. 13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.728** (lower is better). Its 95% predictive intervals cover **90.4%** of exact held-out measurements. The quadrant-deletion test scored 56/56 rows with an equal-quadrant score of 1.860. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 4/4 fitted folds passed sampling checks.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
