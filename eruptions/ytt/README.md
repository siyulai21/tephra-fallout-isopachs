# Younger Toba Tuff (YTT) fallout reconstruction

The accepted dataset contains 46 exact measurements. The regional fits share an observation hull of 17,719,965 km². [Accepted thickness CSV](../../data/eruptions/YTT_tephra.csv) · [Row-level source audit](../../data/audits/YTT/README.md)

## Volumes and geographic prediction

Each volume integrates its method’s median-thickness map. The GP and spline cover the same observation hull. The other three methods cover Earth, including fallout below 0.5 cm. Conditional 95% intervals describe variation in whole-deposit volume across posterior draws.

| Method | Median-map volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| GP ensemble | **1,079** | 760–1.86e+03 | 1.934 | 91.3% |
| Cubic spline | **944** | 798–2e+03 | 2.153 | 82.6% |
| [Directional ensemble](results/directional/README.md) | **900** | 736–3.66e+03 | 1.531 † | 100.0% |
| [Spherical mixture](results/spherical-mixture/README.md) | **1,284** | 1.05e+03–4.87e+04 | 1.756 | 84.8% |
| [Bayesian nested corridors](results/nested-isopachs/README.md) | **1,245** | 771–3e+03 (provisional) | 1.429 † | 97.8% |

Lower scores indicate better predictions in withheld geographic regions. Coverage counts exact withheld measurements. A dagger marks a numerical or geographic validation warning. Provisional intervals retain their fit or integration warnings.

## Full deposits and near-vent detail

The outer line is 0.5 cm. Contours and observation colours double from there. All three methods use the same near-vent window for this eruption. Each curve counts the combined area of all patches at or above a thickness.

![YTT · Directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/directional/deposit.png)

![YTT · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/spherical-mixture/deposit.png)

![YTT · Bayesian nested corridors. Full deposit, near-vent detail and thickness versus square root of area.](results/nested-isopachs/deposit.png)

## Inside the observation hull

![YTT · GP ensemble and cubic spline on a common scale and the full observation hull](results/isopachs.png)

The dashed boundary encloses all accepted sites. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/README.md)

## Evidence and limitations

The principal selection includes the two qualified Maninjau beds and core U1445B at 8 cm. The audit records source qualifications and alternative selections. All six full-data directional fits passed their checks. Two buffered geographic folds carry sampling warnings.

The standard local-support footprint covers 1,196,058 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)

The corridor fit improves buffered predictions and fares worse on withheld quadrants. The full-data fit and volume population checks retained numerical warnings. [Whole-area continuation sensitivity](results/nested-whole-area/README.md)
