# How much does uncertain fan direction change the fit?

Each pair uses the same observations and structure weights. The control fixes fan direction. The principal fit lets it vary across possible deposits. Integrating the pointwise median map gives the first volume. Integrating each complete deposit gives the volume median and conditional interval in the last column.

| Eruption | Accepted observations | Uncertain median map (km³) | Fixed median map (km³) | Complete-deposit median [95%] (km³) |
|---|---:|---:|---:|---:|
| [Aso-4](../../eruptions/aso-4/results/directional/README.md) | 87 | **659.1** | 779.7 | 916.3 [297.9–3,511.5] |
| [B-Tm](../../eruptions/b-tm/results/directional/README.md) | 71 | **33.6** | 34.6 | 37.2 [23.5–77.1] |
| [Ilopango TBJ (Unit G)](../../eruptions/ilopango-tbj/results/directional/README.md) | 56 | **40.8** | 46.7 | 59.4 [30.1–384.8] |
| [Mazama](../../eruptions/mazama/results/directional/README.md) | 45 | **166.1** | 177.0 | 213.5 [138.6–1,265.1] |
| [Minoan](../../eruptions/minoan/results/directional/README.md) | 101 | **28.8** | 32.9 | 39.1 [21.3–138.5] |
| [Tambora](../../eruptions/tambora/results/directional/README.md) | 26 | **95.7** | 107.7 | 117.5 [73.2–203.8] |
| [YTT](../../eruptions/ytt/results/directional/README.md) | 46 | **900.6** | 1,005.6 | 1,272.1 [736.5–3,660.9] |

At B-Tm the source review adds two lake limits and one peat limit to the directly measured thicknesses and bounds. The before-and-after study isolates how those interpretations move the western contours. Each eruption report above gives all eight compass bearings, supported contour fragments and numerical warnings. [B-Tm occurrence study](../../eruptions/b-tm/results/cryptotephra/README.md)

## Predictions in withheld regions

Each score gives equal weight to available folds within an eruption, then equal weight to eruptions. Lower is better. The weights for a target exclude that eruption. The table uses each published fit’s recorded calibration. The passing-only rows show how sampling warnings affect the comparison.

| Deletion design | Eruptions contributing | Fixed | Uncertain azimuth |
|---|---:|---:|---:|
| Geographic, all estimable folds | 7 | 1.582 | 1.580 |
| Geographic, passing folds only | 7 | 1.725 | 1.669 |
| Quadrants, all estimable folds | 7 | 2.040 | 2.014 |
| Quadrants, passing folds only | 6 | 2.025 | 1.960 |

A quadrant with no test observations has no score. The download gives input hashes, fold and row counts, and both scores for every eruption. In the original 68-site B-Tm analysis, only two quadrants could be scored. Adding observations can change which sectors contribute to a pooled comparison. The original six eruptions informed model development. [Paired scores and volumes](case-comparison.csv) · [Original 68-site comparison](https://github.com/siyulai21/tephra-fallout-isopachs/tree/f155753dad2dd4d06e4fa48d3d877242ae8539aa/comparisons/directional)

## Known deposits with a deliberate sampling gap

Fourteen previously specified simulated datasets cover seven designs and two seeds. Two compatible truths on each invisible-sector dataset share all observations but differ by a hidden lobe. The table evaluates false extension and missed fallout on area-weighted whole-Earth nodes, separately for locations with directional exposure of at least 0.5. Each design and seed receives equal weight; compatible truths share that weight. Cells require at least 64 effective union nodes for each of the three mapped decisions.

| Region | Thickness | Resolved datasets | False extension, fixed → uncertain (km²) | Missed area, fixed → uncertain (km²) | Overlap, fixed → uncertain |
|---|---:|---:|---:|---:|---:|
| whole Earth | 1 cm | 14 | 160,917 → 153,346 | 3,538 → 3,861 | 0.797 → 0.805 |
| whole Earth | 10 cm | 14 | 15,000 → 14,022 | 11,646 → 11,796 | 0.787 → 0.788 |
| whole Earth | 100 cm | 12 | 80 → 81 | 6,674 → 6,674 | 0.109 → 0.109 |
| high exposure | 1 cm | 4 | 1,627 → 112 | 0 → 690 | 0.953 → 0.976 |
| high exposure | 10 cm | 4 | 0 → 0 | 6,054 → 6,367 | 0.619 → 0.589 |
| high exposure | 100 cm | 4 | 0 → 0 | 6,228 → 6,228 | 0.000 → 0.000 |

Across the resolved high-exposure tests, the revised 1 cm contour cuts false extension at the cost of some missed area. For 10 cm there was already no false extension in those tests, and shortening the contour misses more true fallout. The stricter 90% exceedance outline misses still more. Nine of the fourteen synthetic datasets had a sampling warning in at least one base structure. All four datasets with resolved high-exposure 1 cm and 10 cm comparisons had such a warning. The passing-only subset therefore provides no resolved test of those high-exposure comparisons. These patterns describe a trade-off under the simulated assumptions. They cannot establish the unseen shape of a real deposit. [False-positive and false-negative areas for every trial and threshold](synthetic-areas.csv) · [True volumes and base-fit warnings](synthetic-trials.json)

## Methods and earlier controls

[Equations and calibration](../../docs/methods/uncertain-azimuth.md) · [Original six-fit identities before the B-Tm revision](verification.json)

- [Fixed-direction ensemble](fixed-ensemble/README.md) examines the earlier weight calibration, geographic prediction and volume coverage on known deposits. Its fixed-direction estimates use different weights from the paired control above.
- [Measured fans and inspected sites](measured-fans/README.md) tests continuation along sampled bearings and links the B-Tm evidence for recognizable beds.
- [Single-structure sensitivities](single-structure/README.md) tests 13 fixed policies, an unresolved proximal cap and independent replications.
