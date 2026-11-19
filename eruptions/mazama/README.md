# Mazama fallout reconstruction

The accepted dataset contains 45 exact measurements. The regional fits share an observation hull of 930,447 km². [Accepted thickness CSV](../../data/eruptions/Mazama_tephra.csv) · [Input provenance](../../data/eruptions/README.md)

## Isopach fits

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at Mazama](results/four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, white dashes show conditional continuation. Solid white segments connect thicknesses bracketed by measured sites in a supported triangle. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **141.64** | 109.26–179.21 | 1.272 | 95.6% |
| Cubic spline | Hull mean **132.61** | 120.25–151.97 | 1.971 | 82.2% |
| [Directional ensemble](results/directional/README.md) | Whole-Earth median map **179.9** · volume median 224.3 | 149.1–1,371.9 | 1.407 | 95.6% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **223.2** | 144.7–3,668.8 | 1.577 † | 91.1% |

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval. † A geographic fit had a numerical warning or incomplete scoring.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **14.6%** of its volume. The directional volume interval comes from complete simulated deposits. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the two whole-Earth maps</summary>

| Directional ensemble | Spherical mixture |
|---|---|
| ![Directional ensemble median thickness over the Earth for Mazama](results/directional/ensemble/world.png) | ![Spherical-mixture median thickness over the Earth for Mazama](results/spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>


## How firmly do the data locate the edge of the deposit?

Letting a poorly constrained fan turn gives a median-map volume of **166.1 km³**. The matched fixed-direction control gives **177.0 km³**. Both use the same training observations and weights. The established six-structure model retains the primary global estimate because predictive performance across withheld directions was mixed.

![Measured and extrapolated isopach segments near the volcano](results/directional/azimuth/median-isopachs.png)

[Directional extent, map probabilities, sector areas and held-out tests](results/directional/azimuth/README.md)

## Evidence and limitations

The calculation uses the supplied acceptance flags. An independent row-by-row source review is still outstanding. Some spherical-mixture prior refits carry numerical warnings, and the broad upper interval reflects unsampled fallout.

The standard local-support footprint covers 172,792 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
