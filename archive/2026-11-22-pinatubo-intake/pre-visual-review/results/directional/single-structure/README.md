# Pinatubo 1991 (Layer C) single-structure directional sensitivity

**6.1–11.2 km³ of bulk fallout** under the common two-scale continuation with an inferred distal thinning law. This is the conditional 95% volume interval. A median **7.9%** of the volume lies inside retained measurement triangles. The remainder depends on extrapolation. **Numerical warning.** The reference failed at least one sampling or integration check.

![Supported isopach fragments and conditional continuation](isopachs.png)

The first map shades the interpolation footprint and shows 57 measured-bracketed line segments. The second map shows the geometric mean of posterior thickness with labelled contours. The panels deliberately use different extents: every observation is visible locally, while the continuation view expands around the 0.1 cm contour. Circles mark exact observations, triangles mark bounds, and the star marks the source. Volume comes from integration of each complete field to the antipode.

![Local thickness and latent uncertainty](local.png)

![Whole-Earth conditional continuation](world.png)

Local detail can cut contours at its window; longitude edges join in the whole-Earth view. Colours saturate below 0.1 cm and above the displayed upper limit. Fine coastlines supply context, not fitting evidence. [Display bounds, finite-grid edge check and posterior subset](map-display.json).

The displayed grid reaches 90.2 cm at a source distance of 0.0 km (background continuation). At the supplied vent it gives 90.2 cm. These are summaries of the fitted map.

## Where the volume comes from

| Region | Conditional 95% volume interval (km³) | Median fraction |
|---|---:|---:|
| Retained measurement triangles | 0.5–0.7 | 7.9% |
| Transition around triangle edges | 0.2–0.2 | 2.5% |
| Remaining area closer than the nearest exact site | 0.1–0.1 | 1.5% |
| Remaining area between the nearest and farthest exact sites | 5.1–9.5 | 85.0% |
| Remaining area beyond the farthest exact site | 0.1–0.9 | 3.0% |

The five regions exhaust the Earth in each draw.

![Volume by compass sector](directions.png)

| Bearing | Exact rows | Bounds | Exact rows in distal half | Footprint km² | Sector 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 1 | 2 | 0 | 0 | 0.0–0.1 |
| NE | 12 | 2 | 4 | 15 | 0.0–0.1 |
| E | 30 | 9 | 11 | 206 | 0.1–0.1 |
| SE | 19 | 10 | 12 | 236 | 0.1–0.1 |
| S | 10 | 7 | 6 | 62 | 0.4–2.2 |
| SW | 33 | 8 | 14 | 4,353 | 1.7–5.1 |
| W | 24 | 12 | 18 | 20,726 | 3.1–4.1 |
| NW | 11 | 4 | 5 | 23 | 0.1–0.3 |

Octants span 45° centred on each compass bearing. Distal half refers to radius at or beyond the median distinct exact-site radius. The largest gap between distal bearings spans **102.6°** around **350.2°**. Measurements at the vent have no bearing. The explicit extra-fan sensitivity gives **6.2–11.2 km³**.

## Unmeasured proximal fallout

The nearest accepted site is 8.7 km from the vent. With a compact positive ash cap filling a source depression, the conditional 95% interval is **6.1–11.2 km³**. It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. The cap-inclusive numerical checks carry a warning. Other unobserved proximal deposits can differ. Its matched geographic score is 1.835, with 3 held-out rows inside a training cap.

Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of 90.2 cm, 0.0 km from the vent in background continuation. With the cap the highest candidate is 0.0 km from the vent. Both peak searches are finite. [Paired volume draws](source-cap.npz).

## Assumptions and validation

| Common policy | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| reference | 6.1–11.2 | review |
| heavy tail | 17.0–36.7 | pass |
| fixed fourth power | 7.0–12.6 | pass |
| exponential | 6.1–11.4 | pass |
| light tail | 6.2–10.6 | pass |
| narrow fans | 5.6–11.8 | pass |
| broad fans | 6.6–9.1 | review |
| possible missing fan | 6.2–11.2 | pass |
| strict footprint | 5.8–10.1 | pass |
| permissive footprint | 5.7–8.8 | pass |
| broader background | 6.2–11.1 | review |
| longer transition | 6.2–11.3 | pass |
| stronger local departures | 6.2–10.8 | pass |

10/13 fits passed the parameter, volume and integration checks. The intervals condition on the fitted field family, sampling locations and deposit identification. Unknown fans, preservation bias and marine redistribution can move the true total outside them.

The matched buffered geographic score is **1.835** (lower is better). Its 95% predictive intervals cover **81.4%** of exact held-out measurements. The quadrant-deletion test scored 194/194 rows with an equal-quadrant score of 3.086. Both tests rebuild the geometry from training rows.

Geographic deletion used four clusters. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 2. Quadrant deletion used four compass quadrants. 3/4 fitted folds passed sampling checks, with a numerical warning in fold 2.

The shared whole-contour audit found 0 policies with a qualifying band out of 3. Its record retains rejected contours, holes and geographic-deletion checks. [Complete contour audit](complete-contour-audit.json)

[Numerical report and fold identities](report.json) · [Every contour decision](isopach-fragments.csv) · [Held-out predictions](heldout.csv) · [Assumption estimates](estimates.csv) · [Joint parameter and node draws](posterior.npz) · [Sector and support mass by chain and draw](directional-mass.npz) · [All sensitivity volume draws](volume-draws.npz)
