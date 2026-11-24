# Mazama climactic fallout

**244 observations enter all five model families.** The common input has 58 measurements, 153 lower bounds and 33 upper bounds. The source ledger reconciles 293 original catalogue entries and 440 additional records, cross-references and controls. The original field evidence remains incomplete at sites identified in the audit.

[Accepted input](../../data/eruptions/Mazama_reviewed.csv) · [Audit assessment](../../data/audits/Mazama/README.md) · [Every locality](../../data/audits/Mazama/LOCALITIES.md) · [Young field sections](../../data/audits/Mazama/YOUNG_AND_FIELD_SECTIONS.md) · [Lake cores and Mount Baker field sections](../../data/audits/Mazama/LAKE_AND_FIELD_RECORDS.md) · [Dated sections](../../data/audits/Mazama/DATED_SECTIONS.md) · [Great Basin and Carp Lake](../../data/audits/Mazama/GREAT_BASIN.md) · [Thicknessless-site derivations](../../data/audits/Mazama/THICKNESSLESS_SITES.md) · [Yellowstone cores](../../data/audits/Mazama/YELLOWSTONE.md) · [Source sensitivities](results/source-audit/README.md)

The target is primary airborne bulk fallout from the climactic eruption, including co-ignimbrite ash. Identified precursor beds, flow deposits and reworked totals are excluded. Preserved primary components and range minima enter once as lower bounds, conditional on their identity and preservation. Green and Tepee Lake positions are corrected; the 850 cm Muir Creek facies-ambiguous value is quarantined. The Williams 46 cm phase ambiguity is retained explicitly and tested by omission.

These global volumes are conditional estimates. Sparse bearings and unsampled regions remain uncertain even where a whole-Earth integral converges. Each family keeps its own interval, prior sensitivity and numerical status. [Shared support decisions](results/support/README.md) and each global extent report preserve rejected contours and unsupported gaps.

## Volumes and geographic prediction

Each volume integrates its method’s median-thickness map. The GP and spline cover the same observation hull. The other three methods cover Earth, including fallout below 0.5 cm. Conditional 95% intervals describe variation in whole-deposit volume across posterior draws.

| Method | Median-map volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| GP ensemble | **409** | 332–835 | 1.166 | 74.1% |
| Cubic spline | **2,138** | 1.13e+03–4.72e+03 | 2.634 | 81.0% |
| [Directional ensemble](results/directional/README.md) | **332** | 257–1.04e+03 (provisional) | 0.813 † | 87.9% |
| [Spherical mixture](results/spherical-mixture/README.md) | **370** | 212–7.17e+03 | 1.000 | 72.4% |
| [Bayesian nested corridors](results/nested-isopachs/README.md) | **271** | 187–724 | 1.011 | 72.4% |

Lower scores indicate better predictions in withheld geographic regions. Coverage counts exact withheld measurements. A dagger marks a numerical or geographic validation warning. Provisional intervals retain their fit or integration warnings.

## Full deposits and near-vent detail

The outer line is 0.5 cm. Contours and observation colours double from there. All three methods use the same near-vent window for this eruption. Each curve counts the combined area of all patches at or above a thickness.

![Mazama · Directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/directional/deposit.png)

![Mazama · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/spherical-mixture/deposit.png)

![Mazama · Bayesian nested corridors. Full deposit, near-vent detail and thickness versus square root of area.](results/nested-isopachs/deposit.png)

## Inside the observation hull

![Mazama · GP ensemble and cubic spline on a common scale and the full observation hull](results/isopachs.png)

The dashed boundary encloses all accepted sites. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/README.md)


## Evidence, reproduction and history

![Mazama evidence coverage](../../data/audits/Mazama/evidence-map.png)

Ten Yellowstone Lake cores contain 4–5 mm of Mazama ash at eight positions. Seven positions enter the principal fit. Two core pairs share positions, and YL16-6A lies at a slump toe. The enlarged map resolves the cores that overlap at the continental scale. [Core identities and bed measurements](../../data/audits/Mazama/YELLOWSTONE.md)

![Yellowstone Lake core locations and ash thicknesses](../../data/audits/Mazama/yellowstone-cores.png)

The fit also includes 8 cm at Overland Lake, 4 cm at Soldier Lake, 1 cm in Great Salt Lake core GSL00-4A-2H-2, and the approximately 4 cm airfall bed in the Lyman Lake composite. [Primary core reports](../../data/audits/Mazama/FOLLOWUP_SEARCH.md) · [Lyman position and stratigraphy](../../data/audits/Mazama/LATE_SOURCE_REVIEW.md)

The nested-model edge test refits the deposit after withholding the most influential outer site and again after withholding its geographic cluster. [Withheld rows, volumes and numerical checks](results/nested-whole-area/edge-deletion/README.md)

Run `tephra-isopachs compare Mazama --output results/new-mazama-run` for the registered comparison. The source-refit commands and publication verification are recorded with the [release](results/source-audit/README.md). Earlier published results and superseded audit trials retain their inputs and hashes in [dated archives](../../archive/2026-11-23-mazama-source-audit/README.md).
