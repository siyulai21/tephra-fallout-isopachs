# Aso-4 fallout reconstruction

The accepted dataset contains 83 exact measurements, 3 lower bounds and 1 upper bound. The regional fits share an observation hull of 3,482,991 km². [Accepted thickness CSV](../../data/eruptions/Aso-4_reviewed.csv) · [Row-level source audit](../../data/audits/Aso-4/README.md)

## Isopach fits

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at Aso-4](results/four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, white dashes show conditional continuation. Solid white segments connect thicknesses bracketed by measured sites in a supported triangle. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **202.65** | 155.87–266.97 | 1.514 | 94.0% |
| Cubic spline | Hull mean **248.15** | 168.43–452.02 | 1.662 | 90.4% |
| [Directional ensemble](results/directional/README.md) | Whole-Earth median map **740.0** · volume median 882.7 | 397.5–2,931.7 | 1.507 | 86.7% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **1,466.0** | 501.0–44,172.9 | 1.632 | 95.2% |

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **3.3%** of its volume. The directional volume interval comes from complete simulated deposits. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the two whole-Earth maps</summary>

| Directional ensemble | Spherical mixture |
|---|---|
| ![Directional ensemble median thickness over the Earth for Aso-4](results/directional/ensemble/world.png) | ![Spherical-mixture median thickness over the Earth for Aso-4](results/spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>


## How firmly do the data locate the edge of the deposit?

Letting a poorly constrained fan turn gives a median-map volume of **659.1 km³**. The matched fixed-direction control gives **779.7 km³**. Both use the same training observations and weights. The established six-structure model retains the primary global estimate because predictive performance across withheld directions was mixed.

![Measured and extrapolated isopach segments near the volcano](results/directional/azimuth/median-isopachs.png)

[Directional extent, map probabilities, sector areas and held-out tests](results/directional/azimuth/README.md)

## Evidence and limitations

The reviewed input corrects two cruise locations and retains the Biwa source conflict and the Hokkaido erosion interpretation. The nearest accepted site is 88 km from the vent. The southeast, south and west have two, two and zero exact measurements. The large spherical-mixture total remains questioned. Omitting all 28 source-labelled marine records raises its median to about 3,184 km³. A regional deletion also makes the spline partial volume unstable. [Source, geometry and influence experiments](../../comparisons/aso-4-audit/README.md)

The standard local-support footprint covers 711,348 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
