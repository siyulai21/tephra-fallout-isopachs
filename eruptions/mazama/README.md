# Mazama source audit and model update

The revised ledger contains 713 evidence records. The current input contains 236 observations, with 57 measured beds, 152 lower bounds and 27 upper bounds. The original field logs supply the Yellowstone, Nevada, Utah and Lyman Lake measurements, Young's mapped sections and two newly located lake beds in the UBC Research Forest. [Audit and sources](../../data/audits/Mazama/README.md) · [Accepted observations](../../data/eruptions/Mazama_reviewed.csv) · [Dated sections](../../data/audits/Mazama/DATED_SECTIONS.md)

[Regional and nested model results on the 236 observations](audit-run/README.md) · [GitHub recalculations](https://github.com/siyulai21/tephra-fallout-isopachs/actions/workflows/mazama-recompute.yml)

## Archived model comparison on 45 measurements

The accepted dataset contains 45 exact measurements. The regional fits share an observation hull of 930,447 km². [Accepted thickness CSV](../../data/eruptions/Mazama_tephra.csv) · [Input provenance](../../data/eruptions/README.md)

## Volumes and geographic prediction

Each volume integrates its method’s median-thickness map. The GP and spline cover the same observation hull. The other three methods cover Earth, including fallout below 0.5 cm. Conditional 95% intervals describe variation in whole-deposit volume across posterior draws.

| Method | Median-map volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| GP ensemble | **130** | 109–179 | 1.272 | 95.6% |
| Cubic spline | **129** | 120–152 | 1.971 | 82.2% |
| [Directional ensemble](results/directional/README.md) | **166** | 138–1.19e+03 (provisional) | 1.366 | 95.6% |
| [Spherical mixture](results/spherical-mixture/README.md) | **171** | 135–914 | 1.559 † | 84.4% |
| [Bayesian nested corridors](results/nested-isopachs/README.md) | **219** | 116–960 | 1.369 | 95.6% |

Lower scores indicate better predictions in withheld geographic regions. Coverage counts exact withheld measurements. A dagger marks a numerical or geographic validation warning. Provisional intervals retain their fit or integration warnings.

## Full deposits and near-vent detail

The outer line is 0.5 cm. Contours and observation colours double from there. All three methods use the same near-vent window for this eruption. Each curve counts the combined area of all patches at or above a thickness.

![Mazama · Directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/directional/deposit.png)

![Mazama · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/spherical-mixture/deposit.png)

![Mazama · Bayesian nested corridors. Full deposit, near-vent detail and thickness versus square root of area.](results/nested-isopachs/deposit.png)

## Inside the observation hull

![Mazama · GP ensemble and cubic spline on a common scale and the full observation hull](results/isopachs.png)

The dashed boundary encloses all accepted sites. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/README.md)

The current global fits use the same audited rows. [What changed in the maps, volumes and validation](../../comparisons/global-extent/README.md) · [Continuation rules](../../docs/methods/finite-spread.md)

## Evidence and limitations

The calculation uses the supplied acceptance flags. An independent row-by-row source review is still outstanding. Some spherical-mixture prior refits carry numerical warnings, and the broad upper interval reflects unsampled fallout.

The standard local-support footprint covers 172,792 km². No complete isopach band passed the support and deletion checks. [Support map, partial volumes and rejected contours](results/support/README.md)

In the [preceding reconstruction](../../archive/2026-11-22-global-extent/README.md), the southwestern sector has no measured sites or bounds. Its median 1 cm footprint changes from 28,185 to 5,771 km². Buffered thickness predictions improve, and withholding whole quadrants slightly favors the broader fit. [Whole-area continuation sensitivity](results/nested-whole-area/README.md)
