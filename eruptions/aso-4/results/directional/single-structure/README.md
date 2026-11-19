# Aso-4 single-structure directional sensitivity

**743.6 [391.6–2,030.3] km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **3.4%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The fragments panel covers every accepted location. The continuation panel expands around the displayed 0.1 cm contour; these panels deliberately have different frames. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. This visual refresh preserves the original posterior and all numerical estimates; [replay checks](../../figure-replay.json) record their hashes.


The fragments panel shades the interpolation footprint and shows 11 measured-bracketed line segments. The continuation panel shows the geometric mean of posterior thickness with labelled contours. Both panels mark exact observations with circles, lower and upper bounds with up and down triangles, and the source with a star. Volume comes from integration of each complete field to the antipode.

The original execution grid reached 83.8 cm at a source distance of 48.9 km (background continuation). At the supplied vent it gives 40.3 cm. These original-grid diagnostics remain in `report.json`; the current display grid and posterior subset are recorded separately in [map-display.json](map-display.json).

## Where the volume comes from

| Region | Median bulk km³ [95% interval] | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 25.3 [20.7–31.7] | 3.4% |
| Transition around triangle edges | 30.5 [25.1–39.1] | 4.2% |
| Remaining area closer than the nearest exact site | 4.9 [2.4–9.9] | 0.7% |
| Remaining area between the nearest and farthest exact sites | 231.8 [172.2–400.7] | 31.4% |
| Remaining area beyond the farthest exact site | 448.4 [141.8–1,597.4] | 60.1% |

The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |
|---|---:|---:|---:|---:|---:|
| N | 8 | 0 | 0 | 11,828 | 50.2 [22.0–243.1] |
| NE | 60 | 4 | 33 | 338,921 | 356.5 [169.5–863.0] |
| E | 11 | 0 | 9 | 358,848 | 201.1 [92.5–652.7] |
| SE | 2 | 0 | 0 | 509 | 47.4 [12.8–392.1] |
| S | 2 | 0 | 0 | 992 | 7.6 [4.2–86.3] |
| SW | 0 | 0 | 0 | 23 | 7.3 [3.9–77.9] |
| W | 0 | 0 | 0 | 19 | 7.3 [3.9–77.9] |
| NW | 0 | 0 | 0 | 208 | 7.3 [3.9–83.3] |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **304.1°** around **237.0°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **827.0 [411.8–2,780.4] km³**.

## Unmeasured proximal fallout

The nearest accepted site is 88.3 km from the vent. A compact positive ash cap filling a source depression adds a median **8.16 km³** and gives **756.2 [401.2–2,042.4] km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.555, with 13 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 86.1 cm, 44.2 km from the vent in transition. With the cap the highest candidate is 16.6 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Median bulk km³ [95% interval] | Numerical checks |
|---|---:|---|
| reference | 743.6 [391.6–2,030.3] | pass |
| heavy tail | 1,213.7 [778.3–2,343.0] | pass |
| fixed fourth power | 882.6 [503.5–1,936.2] | pass |
| exponential | 629.9 [355.8–1,599.6] | pass |
| light tail | 751.7 [426.4–1,868.7] | pass |
| narrow fans | 757.0 [351.0–2,453.1] | pass |
| broad fans | 1,214.3 [478.0–3,171.8] | pass |
| possible missing fan | 827.0 [411.8–2,780.4] | pass |
| strict footprint | 832.7 [397.0–2,688.7] | pass |
| permissive footprint | 693.2 [361.8–1,893.8] | pass |
| broader background | 727.6 [383.9–2,112.7] | pass |
| longer transition | 737.6 [381.9–2,093.1] | pass |
| stronger local departures | 747.1 [383.0–2,071.4] | pass |

The 13 median estimates span 629.9–1,214.3 km³ across named assumptions. 13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.555** (lower is better). Its 95% predictive intervals cover **86.7%** of exact held-out measurements. The quadrant-deletion test scored 87/87 rows with an equal-quadrant score of 1.550. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 3/3 fitted folds passed sampling checks. Unscored folds were 2 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
