# Minoan fallout reconstruction

**Principal Bayesian nested estimate.** The corridor model gives 28.4 km³ of bulk fallout (conditional 95% interval 17.4–67.6), compared with 37.6 km³ under the whole-area continuation. All accepted Plinian and co-ignimbrite fans enter the fit together. Both sets of held-out thicknesses favor the broader continuation. [Corridor map, volume and checks](results/nested-isopachs/README.md) · [Full-compass isopachs](results/nested-isopachs/compass.png)

The accepted dataset contains 92 exact measurements and 9 lower bounds. The regional fits share an observation hull of 760,879 km². [Accepted thickness CSV](../../data/eruptions/Minoan_tephra.csv) · [Row-level source audit](../../data/audits/Minoan/README.md)

## Isopach fits

![Principal nested corridor median isopachs for Minoan](results/nested-isopachs/isopachs.png)

On the nested map, solid contour segments have a local thickness bracket and at least 90% posterior crossing probability. Dashes show conditional continuation. [Corridor estimate, support and uncertainties](results/nested-isopachs/README.md)

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at Minoan](results/four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, solid white segments connect thicknesses bracketed by measured sites in a supported triangle. White dashes on the global maps show conditional isopachs. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **27.21** | 18.21–40.34 | 1.451 | 96.7% |
| Cubic spline | Hull mean **24.78** | 24.79–25.00 | 1.601 | 96.7% |
| [Directional ensemble · uncertain fan direction](results/directional/README.md) | Whole-Earth median map **28.8** · volume median 39.1 | 21.3–138.5 | 1.782 | 94.6% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **40.8** | 24.6–1,989.1 | 1.575 | 94.6% |
| [Bayesian nested · corridor](results/nested-isopachs/README.md) | Whole-Earth volume median **28.4** | 17.4–67.6 | 1.621 | 93.5% |

The nested corridor assigns a median 8.6% of bulk volume to the retained exact-site triangles and 2.7% beyond the farthest exact site. Its median map integrates to 24.9 km³. 4/4 buffered fits passed their numerical checks. [Whole-area sensitivity](results/nested-whole-area/README.md) · [Paired median isopachs](results/nested-isopachs/paired-isopachs.png) · [Five-family volume and geographic prediction plot](results/overview.png)

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **8.5%** of its volume. The directional volume interval comes from complete posterior deposit draws. The spherical mixture reports the median integrated volume across broad and uncertain fan continuations. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the three whole-Earth maps</summary>

| Uncertain fan direction | Spherical mixture | Nested corridor |
|---|---|---|
| ![Directional whole-Earth median](results/directional/azimuth/world.png) | ![Spherical whole-Earth median](results/spherical-mixture/world.png) | ![Nested corridor whole-Earth median](results/nested-isopachs/world.png) |

Each map is centred on the vent longitude and has its own thickness key.

</details>

With the same observations and structure weights, fixing the fan directions gives a median-map volume of **32.9 km³**.

<details>
<summary>View the directional deposit extent and source area</summary>

![Principal uncertain-fan-direction median isopachs for Minoan](results/directional/azimuth/isopachs.png)

</details>

## Evidence and limitations

The input combines all accepted Plinian and co-ignimbrite airborne fallout fans in one deposit. Nine totals reported with a greater-than sign enter as lower bounds. The reported volumes exclude ignimbrite flow deposits.

The spline’s narrow interval conditions on its fitted trend. Its complete-field draws use a coarser integration grid than the expected-volume calculation, which puts the reported mean just below that interval.

The standard local-support footprint covers 41,066 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)
