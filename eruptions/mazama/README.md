# Mazama fallout reconstruction

The accepted dataset contains 45 exact measurements. The regional fits share an observation hull of 930,447 km². [Accepted thickness CSV](../../data/eruptions/Mazama_tephra.csv) · [Input provenance](../../data/eruptions/README.md)

## Bulk fallout volume

| Model | Area | Volume summary (km³) | Conditional 95% volume interval (km³) |
|---|---|---:|---:|
| [Directional ensemble](results/directional/README.md) | Whole Earth | Median-map integral **179.9** · complete-field median 224.3 | 149.1–1,371.9 |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole Earth | Posterior median **223.2** | 144.7–3,668.8 |
| GP ensemble | Observation hull | Expected **141.64** | 109.26–179.21 |
| Cubic spline | Observation hull | Expected **132.61** | 120.25–151.97 |

The directional interval summarizes complete simulated deposits. Measured interpolation triangles hold **14.6%** of the central directional map’s volume. The regional intervals condition on the fitted settings. [Paired regional maps and held-out tests](results/README.md)

![Regional and whole-Earth volumes, with matched geographic prediction scores for Mazama](results/overview.png)

## Evidence and limitations

The calculation uses the supplied acceptance flags. An independent row-by-row source review is still outstanding. Some spherical-mixture prior refits carry numerical warnings, and the broad upper interval reflects unsampled fallout.

The standard local-support footprint covers 172,792 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
