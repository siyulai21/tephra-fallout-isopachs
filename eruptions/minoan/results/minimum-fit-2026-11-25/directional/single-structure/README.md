# Minoan single-structure directional sensitivity

**26.8–147.2 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **6.5%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 32 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 43.9 cm at a source distance of 165.0 km (retained measurement triangles). At the supplied vent it gives 36.0 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 2.5–3.6 | 6.5% |
| Transition around triangle edges | 3.5–4.9 | 9.0% |
| Remaining area closer than the nearest exact site | 0.0–0.1 | 0.1% |
| Remaining area between the nearest and farthest exact sites | 18.6–53.3 | 66.5% |
| Remaining area beyond the farthest exact site | 0.9–92.2 | 17.3% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 1 | 0 | 1 | 0 | 0.5–18.1 |
| NE | 18 | 0 | 12 | 11,790 | 3.0–7.8 |
| E | 38 | 0 | 22 | 7,774 | 9.0–83.6 |
| SE | 25 | 0 | 15 | 16,361 | 5.0–31.1 |
| S | 8 | 0 | 0 | 1,932 | 0.9–15.2 |
| SW | 4 | 0 | 1 | 2,546 | 0.5–6.0 |
| W | 4 | 0 | 0 | 2,455 | 0.3–5.1 |
| NW | 3 | 0 | 0 | 826 | 0.2–5.9 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **110.3°** around **298.5°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **26.6–149.4 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 16.5 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **26.8–147.2 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 1.857, with 10 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 54.0 cm, 160.4 km from the vent in retained measurement triangles. With the cap the highest candidate is 160.4 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 26.8–147.2 | pass |
| heavy tail | 116.5–513.8 | pass |
| fixed fourth power | 37.3–180.7 | pass |
| exponential | 25.0–92.5 | pass |
| light tail | 28.0–136.2 | pass |
| narrow fans | 29.0–184.2 | pass |
| broad fans | 24.2–62.0 | pass |
| possible missing fan | 26.6–149.4 | pass |
| strict footprint | 26.2–156.7 | pass |
| permissive footprint | 26.7–93.7 | pass |
| broader background | 26.5–144.2 | pass |
| longer transition | 26.3–159.6 | pass |
| stronger local departures | 26.5–156.8 | pass |

13/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.857** (lower is better). Its 95% predictive intervals cover **91.1%** of exact held-out measurements. The quadrant-deletion test scored 101/101 rows with an equal-quadrant score of 2.739. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 4/4 fitted folds passed sampling checks.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
