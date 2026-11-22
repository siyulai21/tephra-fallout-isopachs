# Pinatubo 1991 · climactic Layer C

The 15 June 1991 climactic eruption deposited pumice and ash across Luzon and the South China Sea. This case combines the C1 and C2 fall layers on land with their marine equivalents, including co-ignimbrite fallout. Volumes refer to accumulated bulk deposit. The archive also identifies the separate layers A, B and D. [Field stratigraphy](https://pubs.usgs.gov/pinatubo/paladio/index.html) · [Offshore stratigraphy](https://doi.org/10.1007/s00445-003-0306-x)

**The reviewed Layer C input has 208 sites, comprising 150 measurements, 28 lower bounds and 30 upper bounds.** Two readers compared Koyaguchi C1 and C2 and checked 32 entries in Paladio-Melosantos Figure 7. The inventory contains 464 source records. The exact sites beyond 100 km leave a 294° gap in direction. [Accepted observations](../../data/eruptions/Pinatubo_1991_layer_C.csv) · [Koyaguchi review](../../data/audits/Pinatubo-1991/koyaguchi-c1-review.md) · [Figure 7 review and original crops](../../data/audits/Pinatubo-1991/paladio-figure-7-review.md)

The shared model input draws on 172 land-survey records, 34 marine cores or station groups and two airfield bounds. A further 366 entries document layers A, B and D. Wang’s 1999 dissertation and the original SO-132 station ledger were unavailable to this audit. Marine occurrences without a measured thickness or calibrated limit remain in the evidence inventory. [Offshore source coverage](../../data/audits/Pinatubo-1991/marine-check.md) · [Other fall layers](../../data/audits/Pinatubo-1991/other-phases/README.md)

## Volumes and geographic prediction

Each volume integrates its method’s median-thickness map. The GP and spline cover the same observation hull. The other three methods cover Earth, including fallout below 0.5 cm. Conditional 95% intervals describe variation in whole-deposit volume across posterior draws.

| Method | Median-map volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| GP ensemble | **3.45** | 2.97–4.95 | 3.023 | 87.3% |
| Cubic spline | **14.8** | 10.4–1.69e+11 (provisional) | 1.590 | 85.3% |
| [Directional ensemble](results/directional/README.md) | **6.38** | 4.92–10.8 (provisional) | 1.744 † | 88.7% |
| [Spherical mixture](results/spherical-mixture/README.md) | **11.1** | 7.33–20.2 | 0.965 † | 94.7% |
| [Bayesian nested corridors](results/nested-isopachs/README.md) | **6.77** | 5.57–8.09 | 3.342 † | 60.7% |

Lower scores indicate better predictions in withheld geographic regions. Coverage counts exact withheld measurements. A dagger marks a numerical or geographic validation warning. Provisional intervals retain their fit or integration warnings.

## Full deposits and near-vent detail

The outer line is 0.5 cm. Contours and observation colours double from there. All three methods use the same near-vent window for this eruption. Each curve counts the combined area of all patches at or above a thickness.

![Pinatubo 1991 (Layer C) · Directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](results/directional/deposit.png)

![Pinatubo 1991 (Layer C) · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](results/spherical-mixture/deposit.png)

![Pinatubo 1991 (Layer C) · Bayesian nested corridors. Full deposit, near-vent detail and thickness versus square root of area.](results/nested-isopachs/deposit.png)

## Inside the observation hull

![Pinatubo 1991 (Layer C) · GP ensemble and cubic spline on a common scale and the full observation hull](results/isopachs.png)

The dashed boundary encloses all accepted sites. Gaps inside it can still be poorly constrained. [Regional fits and numerical checks](results/README.md)

The current global fits use the same audited rows. [What changed in the maps, volumes and validation](../../comparisons/global-extent/README.md) · [Continuation rules](../../docs/methods/finite-spread.md)

## What changed after the two map reviews

Four revised Koyaguchi C1 readings changed the paired bed totals or their bounds. The Figure 7 review added seven measured thicknesses, five lower bounds and two upper bounds. Two existing Figure 7 measurements also changed. Relative to the archived 194-site fits, the directional median-map volume changed by -2.4%, the spherical volume by +1.6% and the nested volume by +10.3%. All 14 additional sites are within 5 km of an earlier observation. The distal coverage gap is unchanged. [Five volumes before and after review and predictions at each additional site](results/source-checks/figure7-review/README.md) · [Earlier data and fitted maps](../../archive/2026-11-22-pinatubo-intake/pre-visual-review/README.md)

## Check the measurements

- [Check a land reading by eye](../../data/audits/Pinatubo-1991/VISUAL-CHECK.md). Each evidence ID leads to the original pixels, the reading and the accepted CSV row or exclusion.
- [Compare both mapped positions of the same bed](results/source-checks/README.md#position-on-the-two-sublayer-maps). The alternative refits move the 51 combined sublayer observations to their positions on Koyaguchi’s C2 panel.
- [Check the twelve named sections](../../data/audits/Pinatubo-1991/field-check.md). The 22 grain-size sample entries resolve to twelve whole-section measurements.
- [Check a marine station](../../data/audits/Pinatubo-1991/marine-check.md). The measured bulk bed, stratigraphic subdivisions, cruise locator and repeat visits are kept together.
- [Inspect the other fall layers](../../data/audits/Pinatubo-1991/other-phases/README.md). The preclimactic A and B and postclimactic D measurements retain their original units, source pixels and location qualifications.
- [Check the distal reports](../../data/audits/Pinatubo-1991/distal-reports.md). The audit covers nineteen mapped localities and contemporary field reports, including the proximal 1–2 m areal average.

## Where the observations change the estimate

With all 34 offshore records withheld, the land-only nested fit covers 0 of 19 measured seabeds with its 95% predictive intervals. Its median map integrates to 0.70 km³ and its numerical checks give a warning. With the offshore data included, the principal nested estimate is 8.36 km³. Giving the nested model both endpoints of the nine marine replicate ranges yields 5.20 km³ with a conditional 95% interval of 4.55–7.21 km³. This refit passes its numerical checks. [Offshore holdout, interval fits and station predictions](results/source-checks/README.md)

The GP and spline integrate over the same 279,806 km² hull. At SO-114-13 the measured marine bed spans 1–2 mm. Under the shared lower-bound treatment, the spline median reaches 37.6 m at this station and its upper volume interval fails the integration check. Using the upper endpoints of the nine replicate ranges as upper bounds changes its regional volume from 14.82 to 5.04 km³. The regional spline is highly sensitive to how these ranges enter the fit. [Endpoint and location checks](results/source-checks/README.md#regional-sensitivity-to-the-upper-endpoints)

In the [earlier fitted comparison](../../archive/2026-11-22-global-extent/README.md), the uncertain-direction ensemble covers 134 of 150 scored measurements with its geographic 95% prediction intervals, out of 150 exact sites overall. 4 of 6 full-data members pass sampling checks. The spherical family places a median 4.0% of its mass inside the screened measurement footprint and 94.3% in intervening gaps and sectors. Those fractions are calculated across whole-deposit draws. The spherical 0.5 cm contour extends well south of the offshore cores, where its reach depends on the assumed fallout geometry. [Directional holdouts](results/directional/azimuth/heldout.csv) · [Spherical mass by region](results/spherical-mixture/README.md)

In the [earlier fitted comparison](../../archive/2026-11-22-global-extent/README.md), the spherical baseline passes its sampling checks with 0 divergent transitions. Across the eight fixed prior assumptions, 6 fits pass their numerical checks. The combined range of the eight conditional intervals is 6.52–171.56 km³. This sensitivity range has no joint probability level. Only one of the four geographic training fits passes sampling checks. [Prior choices and diagnostics](results/spherical-mixture/report.json)

In the [earlier fitted comparison](../../archive/2026-11-22-global-extent/README.md), the nested corridor full fit passes its population and integration checks. 3 of its four geographic training fits pass. The geographic prediction intervals cover 93 of 150 withheld measurements. None of the tested support policies accepts a complete nested contour band. Beyond 100 km the measured sites leave a 294° unsampled arc. Whole-Earth totals depend on continuation across those bearings. [Observed geographic coverage](../../data/audits/Pinatubo-1991/README.md#geographic-coverage) · [Nested support and extrapolation](results/nested-isopachs/README.md) · [Whole-area sensitivity to withholding the outermost influential core](results/nested-whole-area/edge-deletion/README.md)

Wiesner and colleagues reported 5.5 km³ for the 15 June airfall deposit. Earlier thinning fits ranged from 1.7 to 4.4 km³ as distal extrapolation assumptions varied. Those estimates and the five fits here share field evidence. [Later land and marine estimate](https://doi.org/10.1007/s00445-003-0306-x) · [Earlier thinning scenarios](../../data/sources/pinatubo-1991/context/paladio-published-volume-comparisons.csv)

## Reproduce the case

Run these commands from the repository root. The comparison runs the two regional families and all three whole-Earth families with their standard settings, prior sensitivities, support screens and geographic holdouts.

```sh
python -m benchmarks.pinatubo
python -m src.empirical_isopach.cli compare pinatubo-1991-layer-c --output results/pinatubo-1991-replay
python -m benchmarks.pinatubo_sensitivity --output results/pinatubo-source-tests
python -m benchmarks.nested_edge_deletion --case data/eruptions/Pinatubo_1991_layer_C.csv --fitted results/pinatubo-1991-replay/nested-whole-area --output results/pinatubo-edge-deletion
python -m benchmarks.pinatubo_field_review --output results/pinatubo-new-sites.csv --volume-output results/pinatubo-review-volumes.csv
```

The history preserves the 137-site development case, the 193-site fits preceding the SO-114-23 recovery, the five 194-site fits from before either visual review, and the Koyaguchi-only data checkpoint. [Intake history and held-out core](../../archive/2026-11-22-pinatubo-intake/README.md) · [Original sources and checksums](../../data/sources/pinatubo-1991/README.md)
