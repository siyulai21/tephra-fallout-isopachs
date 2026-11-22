# Tambora fallout reconstruction

The accepted dataset contains 26 exact measurements. The regional fits share an observation hull of 2,980,287 km². [Accepted thickness CSV](../../data/eruptions/Tambora_tephra.csv) · [Input provenance](../../data/eruptions/README.md)

## Volumes and geographic prediction

Each volume integrates its method’s median-thickness map. The GP and spline cover the same observation hull. The other three methods cover Earth, including fallout below 0.5 cm. Conditional 95% intervals describe variation in whole-deposit volume across posterior draws.

| Method | Median-map volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| GP ensemble | **91.1** | 66.6–143 | 1.620 | 96.2% |
| Cubic spline | **93.5** | 79.5–155 | 1.548 | 96.2% |
| [Directional ensemble](results/directional/README.md) | **95.7** | 73.2–204 (provisional) | 2.066 † | 84.6% |
| [Spherical mixture](results/spherical-mixture/README.md) | **126** | 101–920 | 1.477 | 100.0% |
| [Bayesian nested corridors](results/nested-isopachs/README.md) | **94.9** | 62.5–238 | 1.646 | 100.0% |

Lower scores indicate better predictions in withheld geographic regions. Coverage counts exact withheld measurements. A dagger marks a numerical or geographic validation warning. Provisional intervals retain their fit or integration warnings.

## Full deposits and near-vent detail

The outer line is 0.5 cm. Contours and observation colours double from there. All three methods use the same near-vent window for this eruption. Each curve counts the combined area of all patches at or above a thickness.

![Tambora · Directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/directional/deposit.png)

![Tambora · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/spherical-mixture/deposit.png)

![Tambora · Bayesian nested corridors. Full deposit, near-vent detail and thickness versus square root of area.](results/nested-isopachs/deposit.png)

## Inside the observation hull

![Tambora · GP ensemble and cubic spline on a common scale and the full observation hull](results/isopachs.png)

The dashed boundary encloses all accepted sites. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/README.md)

## Evidence and limitations

The calculation uses the supplied acceptance flags. An independent row-by-row source review is still outstanding. The median-map integration checks passed, but several held-out fits also carry warnings.

The standard local-support footprint covers 49,323 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)

Both geographic and whole-quadrant holdouts predict the real observations better under the broader continuation. [Whole-area continuation sensitivity](results/nested-whole-area/README.md)
