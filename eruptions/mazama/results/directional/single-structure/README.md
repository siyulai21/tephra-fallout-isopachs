# Mazama single-structure directional sensitivity

**150.6–370.2 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. These endpoints give the conditional 95% volume interval. A median **13.5%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The fragments panel covers every accepted location. The continuation panel expands around the displayed 0.1 cm contour; these panels deliberately have different frames. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. This visual refresh preserves the original posterior and all numerical estimates; [replay checks](../../figure-replay.json) record their hashes.


The fragments panel shades the interpolation footprint and shows 17 measured-bracketed line segments. The continuation panel shows the geometric mean of posterior thickness with labelled contours. Both panels mark exact observations with circles, lower and upper bounds with up and down triangles, and the source with a star. Volume comes from integration of each complete field to the antipode.

The original execution grid reached 516.2 cm at a source distance of 30.6 km (transition). At the supplied vent it gives 419.3 cm. These original-grid diagnostics remain in `report.json`; the current display grid and posterior subset are recorded separately in [map-display.json](map-display.json).

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 21.6–34.6 | 13.5% |
| Transition around triangle edges | 46.6–88.1 | 31.6% |
| Remaining area closer than the nearest exact site | 0.0–0.0 | 0.0% |
| Remaining area between the nearest and farthest exact sites | 61.6–146.9 | 42.5% |
| Remaining area beyond the farthest exact site | 5.5–136.0 | 10.9% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 21 | 0 | 11 | 87,683 | 27.9–44.2 |
| NE | 19 | 0 | 11 | 84,375 | 73.2–244.8 |
| E | 3 | 0 | 1 | 510 | 15.4–59.4 |
| SE | 0 | 0 | 0 | 0 | 1.9–9.3 |
| S | 0 | 0 | 0 | 0 | 1.8–9.3 |
| SW | 0 | 0 | 0 | 0 | 2.1–10.6 |
| W | 0 | 0 | 0 | 0 | 2.8–12.7 |
| NW | 2 | 0 | 0 | 224 | 4.8–16.8 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **270.3°** around **215.7°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives 168.2–1,741.9 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 24.5 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **153.5–376.0 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.451, with 8 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 547.9 cm, 24.5 km from the vent in retained measurement triangles. With the cap the highest candidate is 1.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 150.6–370.2 | pass |
| heavy tail | 403.3–1,217.0 | pass |
| fixed fourth power | 184.9–550.4 | pass |
| exponential | 148.8–346.0 | pass |
| light tail | 157.7–405.0 | pass |
| narrow fans | 153.3–464.8 | pass |
| broad fans | 133.0–258.8 | pass |
| possible missing fan | 168.2–1,741.9 | pass |
| strict footprint | 149.0–358.1 | pass |
| permissive footprint | 147.7–271.2 | pass |
| broader background | 149.4–361.0 | pass |
| longer transition | 162.8–385.0 | pass |
| stronger local departures | 160.1–387.0 | pass |

13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.451** (lower is better). Its 95% predictive intervals cover **91.1%** of exact held-out measurements. The quadrant-deletion test scored 45/45 rows with an equal-quadrant score of 2.070. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 2/3 fitted folds passed sampling checks, with a numerical warning in fold 0. Unscored folds were 2 (empty).

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
