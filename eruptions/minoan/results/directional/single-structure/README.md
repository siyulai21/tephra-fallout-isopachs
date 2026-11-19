# Minoan single-structure directional sensitivity

**42.5 [26.2–126.9] km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **6.4%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The fragments panel covers every accepted location. The continuation panel expands around the displayed 0.1 cm contour; these panels deliberately have different frames. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. This visual refresh preserves the original posterior and all numerical estimates; [replay checks](../../figure-replay.json) record their hashes.


The fragments panel shades the interpolation footprint and shows 27 measured-bracketed line segments. The continuation panel shows the geometric mean of posterior thickness with labelled contours. Both panels mark exact observations with circles, lower and upper bounds with up and down triangles, and the source with a star. Volume comes from integration of each complete field to the antipode.

The original execution grid reached 67.7 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 67.7 cm. These original-grid diagnostics remain in `report.json`; the current display grid and posterior subset are recorded separately in [map-display.json](map-display.json).

## Where the volume comes from

| Region | Median bulk km³ [95% interval] | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 2.7 [2.2–3.4] | 6.4% |
| Transition around triangle edges | 4.0 [3.3–4.8] | 9.4% |
| Remaining area closer than the nearest exact site | 0.8 [0.4–2.4] | 1.9% |
| Remaining area between the nearest and farthest exact sites | 28.5 [18.0–51.5] | 67.4% |
| Remaining area beyond the farthest exact site | 5.7 [0.5–72.3] | 13.4% |

The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |
|---|---:|---:|---:|---:|---:|
| N | 1 | 0 | 1 | 0 | 2.0 [0.7–16.8] |
| NE | 18 | 0 | 12 | 11,746 | 4.7 [3.1–7.5] |
| E | 34 | 4 | 20 | 7,535 | 18.0 [8.4–67.7] |
| SE | 25 | 0 | 13 | 16,361 | 8.6 [4.8–29.7] |
| S | 8 | 0 | 0 | 1,536 | 2.3 [0.9–11.7] |
| SW | 3 | 1 | 0 | 2,129 | 1.5 [0.6–6.8] |
| W | 2 | 2 | 0 | 1,759 | 1.2 [0.5–6.1] |
| NW | 1 | 2 | 0 | 0 | 0.8 [0.2–4.3] |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **201.7°** around **252.9°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **42.8 [26.6–127.7] km³**.

## Unmeasured proximal fallout

The nearest accepted site is 16.5 km from the vent. A compact positive ash cap filling a source depression adds a median **0.00 km³** and gives **42.5 [26.2–126.9] km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.732, with 10 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 67.7 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Median bulk km³ [95% interval] | Numerical checks |
|---|---:|---|
| reference | 42.5 [26.2–126.9] | pass |
| heavy tail | 189.6 [109.0–455.5] | pass |
| fixed fourth power | 59.1 [34.4–140.0] | pass |
| exponential | 37.5 [24.8–82.0] | pass |
| light tail | 42.5 [27.2–103.4] | pass |
| narrow fans | 49.5 [28.4–151.2] | pass |
| broad fans | 35.4 [25.7–67.6] | pass |
| possible missing fan | 42.8 [26.6–127.7] | pass |
| strict footprint | 41.0 [25.5–120.2] | pass |
| permissive footprint | 40.7 [26.6–87.5] | pass |
| broader background | 42.7 [26.6–121.8] | pass |
| longer transition | 43.1 [26.4–131.3] | pass |
| stronger local departures | 42.4 [26.2–140.5] | pass |

The 13 median estimates span 35.4–189.6 km³ across named assumptions. 13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.732** (lower is better). Its 95% predictive intervals cover **91.3%** of exact held-out measurements. The quadrant-deletion test scored 101/101 rows with an equal-quadrant score of 2.502. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 4/4 fitted folds passed sampling checks.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
