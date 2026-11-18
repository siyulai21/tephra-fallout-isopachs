# YTT directional fallout estimate

**1,170.3 [848.9–1,926.9] km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. The brackets give the conditional 95% credible interval. A median **16.9%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The left map shades the interpolation footprint and shows 25 measured-bracketed line segments. The right map shows the geometric mean of posterior thickness with dashed contours. Both panels mark exact observations with circles, lower and upper bounds with up and down triangles, and the source with a star. Volume comes from integration of each complete field to the antipode.

The displayed grid reaches 35.1 cm at a source distance of 488.3 km (retained measurement triangles). At the supplied vent it gives 27.3 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Median bulk km³ [95% interval] | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 194.3 [142.8–269.5] | 16.9% |
| Transition around triangle edges | 117.1 [90.5–158.4] | 10.1% |
| Remaining area closer than the nearest exact site | 1.6 [0.8–5.1] | 0.1% |
| Remaining area between the nearest and farthest exact sites | 797.1 [539.8–1,350.5] | 67.7% |
| Remaining area beyond the farthest exact site | 46.3 [5.9–313.9] | 4.0% |

The five regions exhaust the Earth in each draw. Their separately reported medians can sum to a different value from the median total.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Median sector km³ [95% interval] |
|---|---:|---:|---:|---:|---:|
| N | 0 | 0 | 0 | 49,324 | 156.9 [60.4–364.2] |
| NE | 5 | 0 | 0 | 109,743 | 76.2 [42.4–179.9] |
| E | 1 | 0 | 0 | 25,124 | 62.8 [29.9–123.7] |
| SE | 2 | 0 | 0 | 42,356 | 46.5 [18.8–223.8] |
| S | 1 | 0 | 0 | 28,939 | 47.1 [17.8–226.5] |
| SW | 0 | 0 | 0 | 30,360 | 10.2 [5.1–75.6] |
| W | 11 | 0 | 7 | 523,184 | 286.2 [199.5–475.1] |
| NW | 26 | 0 | 16 | 387,032 | 434.3 [300.2–727.2] |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **292.7°** around **118.4°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **1,541.3 [944.6–7,366.8] km³**.

## Unmeasured proximal fallout

The nearest accepted site is 362.0 km from the vent. A compact positive ash cap filling a source depression adds a median **90.49 km³** and gives **1,336.6 [947.2–2,615.5] km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.811, with 10 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 35.4 cm, 507.3 km from the vent in retained measurement triangles. With the cap the highest candidate is 7.5 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Median bulk km³ [95% interval] | Numerical checks |
|---|---:|---|
| reference | 1,170.3 [848.9–1,926.9] | pass |
| heavy tail | 2,687.8 [1,730.5–4,521.4] | pass |
| fixed fourth power | 1,549.2 [1,032.4–2,718.9] | pass |
| exponential | 1,145.2 [832.7–1,909.6] | pass |
| light tail | 1,263.0 [890.3–2,152.9] | pass |
| narrow fans | 909.4 [696.0–1,445.3] | pass |
| broad fans | 1,372.7 [1,080.9–1,906.0] | pass |
| possible missing fan | 1,541.3 [944.6–7,366.8] | pass |
| strict footprint | 1,185.6 [854.2–2,007.5] | pass |
| permissive footprint | 1,148.6 [870.1–1,695.4] | pass |
| broader background | 1,233.2 [871.1–2,067.1] | pass |
| longer transition | 1,169.9 [846.5–1,960.6] | pass |
| stronger local departures | 1,169.3 [846.3–2,048.9] | pass |

The 13 median estimates span 909.4–2,687.8 km³ across named assumptions. 13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.811** (lower is better). Its 95% predictive intervals cover **91.3%** of exact held-out measurements. The quadrant-deletion test scored 46/46 rows with an equal-quadrant score of 3.782. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 4/4 fitted folds passed sampling checks.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
