# Pinatubo 1991 · climactic Layer C single-structure directional sensitivity

**5.4–7.6 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **15.2%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. The reference passed the sampling and integration checks.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 52 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 94.9 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 94.9 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 0.8–1.1 | 15.2% |
| Transition around triangle edges | 0.2–0.2 | 2.9% |
| Remaining area closer than the nearest exact site | 0.1–0.2 | 1.9% |
| Remaining area between the nearest and farthest exact sites | 4.1–6.0 | 77.5% |
| Remaining area beyond the farthest exact site | 0.1–0.3 | 2.4% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 1 | 2 | 0 | 0 | 0.0–0.1 |
| NE | 12 | 9 | 4 | 15 | 0.0–0.1 |
| E | 32 | 18 | 11 | 187 | 0.1–0.1 |
| SE | 22 | 11 | 15 | 170 | 0.1–0.1 |
| S | 11 | 8 | 7 | 44 | 0.4–1.1 |
| SW | 35 | 13 | 15 | 13,968 | 1.8–3.3 |
| W | 20 | 20 | 15 | 17,735 | 2.3–2.9 |
| NW | 14 | 7 | 7 | 1,025 | 0.3–0.6 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **92.8°** around **355.1°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **5.4–7.6 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 8.7 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **5.4–7.6 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks passed. Other unobserved proximal deposits can differ. Its matched geographic score is 2.072, with 8 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 94.9 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 5.4–7.6 | pass |
| heavy tail | 15.1–27.6 | pass |
| fixed fourth power | 6.4–9.8 | pass |
| exponential | 5.3–7.5 | pass |
| light tail | 5.6–8.1 | pass |
| narrow fans | 5.3–7.7 | review |
| broad fans | 5.8–7.7 | pass |
| possible missing fan | 5.4–7.6 | pass |
| strict footprint | 5.5–7.5 | pass |
| permissive footprint | 5.4–7.5 | review |
| broader background | 5.4–7.6 | review |
| longer transition | 5.4–7.6 | review |
| stronger local departures | 5.4–7.7 | pass |

9/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **2.072** (lower is better). Its 95% predictive intervals cover **88.4%** of exact held-out measurements. The quadrant-deletion test scored 235/235 rows with an equal-quadrant score of 8.789. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 4/4 fitted folds passed sampling checks. Quadrant deletion used four compass quadrants. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 2.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
