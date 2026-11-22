# Tambora fallout reconstruction

**Principal Bayesian nested estimate.** The corridor model gives 112.7 km³ of bulk fallout (conditional 95% interval 62.5–237.7), compared with 146.9 km³ under the whole-area continuation. Both geographic and whole-quadrant holdouts predict the real observations better under the broader continuation. [Corridor map, volume and checks](results/nested-isopachs/README.md) · [Full-compass isopachs](results/nested-isopachs/compass.png)

The accepted dataset contains 26 exact measurements. The regional fits share an observation hull of 2,980,287 km². [Accepted thickness CSV](../../data/eruptions/Tambora_tephra.csv) · [Input provenance](../../data/eruptions/README.md)

## Isopach fits

![Principal nested corridor median isopachs for Tambora](results/nested-isopachs/isopachs.png)

On the nested map, solid contour segments have a local thickness bracket and at least 90% posterior crossing probability. Dashes show conditional continuation. [Corridor estimate, support and uncertainties](results/nested-isopachs/README.md)

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at Tambora](results/four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, solid white segments connect thicknesses bracketed by measured sites in a supported triangle. White dashes on the global maps show conditional isopachs. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **97.91** | 66.61–142.60 | 1.620 | 96.2% |
| Cubic spline | Hull mean **107.69** | 79.51–155.23 | 1.548 | 96.2% |
| [Directional ensemble · uncertain fan direction](results/directional/README.md) | Whole-Earth median map **95.7** · volume median 117.5 | 73.2–203.8 | 2.066 † | 84.6% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **162.3** | 100.8–920.1 | 1.477 | 100.0% |
| [Bayesian nested · corridor](results/nested-isopachs/README.md) | Whole-Earth volume median **112.7** | 62.5–237.7 | 1.646 | 100.0% |

The nested corridor assigns a median 10.1% of bulk volume to the retained exact-site triangles and 1.9% beyond the farthest exact site. Its median map integrates to 94.9 km³. 4/4 buffered fits passed their numerical checks. [Whole-area sensitivity](results/nested-whole-area/README.md) · [Paired median isopachs](results/nested-isopachs/paired-isopachs.png) · [Five-family volume and geographic prediction plot](results/overview.png)

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval. † A geographic fit had a numerical warning or incomplete scoring.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **11.2%** of its volume. The directional volume interval comes from complete posterior deposit draws. The spherical mixture reports the median integrated volume across broad and uncertain fan continuations. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the three whole-Earth maps</summary>

| Uncertain fan direction | Spherical mixture | Nested corridor |
|---|---|---|
| ![Directional whole-Earth median](results/directional/azimuth/world.png) | ![Spherical whole-Earth median](results/spherical-mixture/world.png) | ![Nested corridor whole-Earth median](results/nested-isopachs/world.png) |

Each map is centred on the vent longitude and has its own thickness key.

</details>

With the same observations and structure weights, fixing the fan directions gives a median-map volume of **107.7 km³**.

4 of six full-data structures carry sampling warnings.

<details>
<summary>View the directional deposit extent and source area</summary>

![Principal uncertain-fan-direction median isopachs for Tambora](results/directional/azimuth/isopachs.png)

</details>

## Evidence and limitations

The calculation uses the supplied acceptance flags. An independent row-by-row source review is still outstanding. The median-map integration checks passed, but several held-out fits also carry warnings.

The standard local-support footprint covers 49,323 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
