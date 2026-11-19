# Younger Toba Tuff (YTT) fallout reconstruction

The accepted dataset contains 46 exact measurements. The regional fits share an observation hull of 17,719,965 km². [Accepted thickness CSV](../../data/eruptions/YTT_tephra.csv) · [Row-level source audit](../../data/audits/YTT/README.md)

## Isopach fits

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at YTT](results/four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, white dashes show conditional continuation. Solid white segments connect thicknesses bracketed by measured sites in a supported triangle. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **1,196.14** | 759.73–1,861.41 | 1.934 | 91.3% |
| Cubic spline | Hull mean **1,203.01** | 797.96–1,998.78 | 2.153 | 82.6% |
| [Directional ensemble](results/directional/README.md) | Whole-Earth median map **985.7** · volume median 1,274.1 | 765.6–3,514.0 | 1.777 † | 93.5% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **1,674.2** | 1,203.6–54,906.9 | 1.777 | 84.8% |

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval. † A geographic fit had a numerical warning or incomplete scoring.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **19.7%** of its volume. The directional volume interval comes from complete simulated deposits. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the two whole-Earth maps</summary>

| Directional ensemble | Spherical mixture |
|---|---|
| ![Directional ensemble median thickness over the Earth for YTT](results/directional/ensemble/world.png) | ![Spherical-mixture median thickness over the Earth for YTT](results/spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>


## How firmly do the data locate the edge of the deposit?

Letting a poorly constrained fan turn gives a median-map volume of **900.6 km³**. The matched fixed-direction control gives **1,005.6 km³**. Both use the same training observations and weights. The established six-structure model retains the primary global estimate because predictive performance across withheld directions was mixed.

![Measured and extrapolated isopach segments near the volcano](results/directional/azimuth/median-isopachs.png)

[Directional extent, map probabilities, sector areas and held-out tests](results/directional/azimuth/README.md)

## Evidence and limitations

The principal selection includes the two qualified Maninjau beds and core U1445B at 8 cm. The audit records source qualifications and alternative selections. All six full-data directional fits passed their checks. Two buffered geographic folds carry sampling warnings.

The standard local-support footprint covers 1,196,058 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
