# Pinatubo 1991 Layer C single-structure directional sensitivity

**5.5 [4.6–7.6] km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **9.4%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. **Numerical warning.** The reference failed at least one sampling or integration check.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 81 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 103.4 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 103.4 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Median bulk km³ [95% interval] | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 0.5 [0.4–0.6] | 9.4% |
| Transition around triangle edges | 0.2 [0.2–0.2] | 3.7% |
| Remaining area closer than the nearest exact site | 0.1 [0.1–0.2] | 2.2% |
| Remaining area between the nearest and farthest exact sites | 4.5 [3.7–6.4] | 82.3% |
| Remaining area beyond the farthest exact site | 0.1 [0.1–0.3] | 2.1% |

The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |
|---|---:|---:|---:|---:|---:|
| N | 1 | 2 | 0 | 12 | 0.0 [0.0–0.1] |
| NE | 8 | 2 | 2 | 139 | 0.1 [0.0–0.1] |
| E | 28 | 8 | 11 | 229 | 0.1 [0.1–0.1] |
| SE | 17 | 7 | 11 | 159 | 0.1 [0.1–0.1] |
| S | 7 | 6 | 5 | 160 | 0.5 [0.2–1.3] |
| SW | 20 | 4 | 8 | 17,851 | 2.0 [1.4–3.3] |
| W | 14 | 4 | 10 | 77,369 | 2.6 [2.2–3.0] |
| NW | 7 | 2 | 4 | 40 | 0.1 [0.1–0.2] |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **102.6°** around **350.2°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **5.5 [4.6–7.6] km³**.

## Unmeasured proximal fallout

The nearest accepted site is 8.7 km from the vent. A compact positive ash cap filling a source depression adds a median **0.00 km³** and gives **5.5 [4.6–7.6] km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks carry a warning. Other unobserved proximal deposits can differ. Its matched geographic score is 14.917, with 9 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 103.4 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Median bulk km³ [95% interval] | Numerical checks |
|---|---:|---|
| reference | 5.5 [4.6–7.6] | review |
| heavy tail | 13.4 [9.8–21.4] | pass |
| fixed fourth power | 5.7 [4.5–8.0] | pass |
| exponential | 5.6 [4.7–7.8] | pass |
| light tail | 5.3 [4.4–7.3] | pass |
| narrow fans | 4.9 [4.1–7.7] | review |
| broad fans | 7.4 [6.2–9.1] | pass |
| possible missing fan | 5.5 [4.6–7.6] | pass |
| strict footprint | 5.6 [4.6–8.0] | review |
| permissive footprint | 5.4 [4.5–7.5] | pass |
| broader background | 5.5 [4.6–7.7] | review |
| longer transition | 5.5 [4.6–7.6] | review |
| stronger local departures | 5.5 [4.6–7.7] | review |

The 13 median estimates span 4.9–13.4 km³ across named assumptions. 7/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **14.917** (lower is better). Its 95% predictive intervals cover **77.5%** of exact held-out measurements. The quadrant-deletion test scored 137/137 rows with an equal-quadrant score of 3.493. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 1. Quadrant deletion used four compass quadrants. 2/4 fitted folds passed sampling checks, with a numerical warning in fold 0, 3.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
