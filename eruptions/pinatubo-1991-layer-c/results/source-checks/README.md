# Which source decisions change the Pinatubo estimate?

The nested-corridor tests change one source selection or reading rule at a time. The volume column integrates the median-thickness field over Earth. The interval describes coherent whole-deposit draws under that fit.

| Fit | Training sites | Volume (km³) | Conditional 95% (km³) | Exact withheld beds within 95% | Numerical checks |
|---|---:|---:|---:|---:|---|
| [Common five-family input](../nested-isopachs/README.md) | 194 | 7.58 | 5.70–9.08 | — | Passed |
| [Withhold every offshore thickness and bound](marine-holdout/result.json) | 160 | 0.70 | 0.61–1.56 | [0/19](marine-holdout/heldout.csv) | Warning · parent population check |
| [Withhold the numerical land-map readings](tabulated-thickness/result.json) | 48 | 7.64 | 6.55–10.43 | [95/109](tabulated-thickness/heldout.csv) | Warning · importance weights |
| [Use both observed endpoints of nine marine station ranges](marine-intervals/result.json) | 194 | 5.16 | 4.44–7.06 | — | Passed |
| [Use the C2 map coordinates for the same 51 paired beds](paired-map-coordinates/result.json) | 194 | 7.76 | 6.54–9.23 | — | Warning · parent population check |

The land-only model predicts the 34 marine records without seeing any of them. The later offshore cruises document a secondary thickness maximum. Their measurements change both the estimated reach and the total. [Predictions at every withheld marine station](marine-holdout/heldout.csv) · [Original offshore survey](https://doi.org/10.1007/s00445-003-0306-x)

The second test keeps 48 published section, core and text readings, including named sections positioned from maps. It withholds 146 land markers whose thickness was transcribed from map figures. The archived site table retains their reading and predictive error. [Withheld land markers](tabulated-thickness/heldout.csv)

At nine marine stations Hess and colleagues report replicate thickness ranges. All five principal fits can use the minimum as a lower bound. The two-sided nested test represents the measured range as an interval at each station. The two endpoints also supply a separate check on both regional fits. [Published ranges](../../../../data/sources/pinatubo-1991/hess-table-1.csv) · [Interval input](marine-intervals/interval-input.csv) · [Median predictions from all five fits at every station](marine-intervals/fitted-bed-thickness.csv)

## Regional sensitivity to the upper endpoints

These two regional fits keep every site and replace the nine lower limits with the corresponding measured upper endpoints as upper limits. Using a single side of a finite range discards information. This controlled change exposes how much either regional fit depends on which side is retained.

| Observation-hull median map | Shared lower endpoints (km³) | Alternative upper endpoints (km³) |
|---|---:|---:|
| GP ensemble | 3.50 | 3.07 |
| Cubic spline | 13.17 | 5.30 |

At SO-114-13 (H01-16), the source gives 1–2 mm, or 0.1–0.2 cm. Under the shared lower-bound rule the spline median at that core is 24.1 m and the GP median is 0.49 cm. A lower bound alone permits much thicker predictions. The spline’s upper draw interval fails its regional integration check. [Regional field checks](regional-median-field-checks.json) · [Upper-endpoint regional report](regional-upper-endpoints/regional/README.md)

## Independent cruise core

The 1.0 cm reading at SO-114-23 became available after the archived 193-site audit. Its preceding fits predicted 0.89 cm (spline), 1.54 cm (GP) and 3.78 cm (nested corridor). Applying the 15 independently measured chart-position discrepancies at the core gave 0.70–0.94, 1.45–1.57 and 3.58–3.84 cm respectively. Those ranges measure sensitivity to position. This check concerns one additional core. All principal fits include it once. [Numerical receipt](cruise-core-holdout/result.json) · [Training evidence and replay](../../../../archive/2026-11-22-pinatubo-intake/README.md) · [Original core verification](../../../../data/audits/Pinatubo-1991/marine-check.md#südmeer-core-23)

Reproduce the source refits with `python -m benchmarks.pinatubo_sensitivity --output results/pinatubo-source-tests`. Each receipt records its training rows, input hash, fit diagnostics and integration checks. Rebuild the station comparison with `python -m benchmarks.pinatubo_ranges --output results/pinatubo-station-predictions.csv`.

## Position on the two sublayer maps

Koyaguchi printed the same 51 accepted stations on the C1 and C2 panels. Independently registered positions differ by a median 0.16 km and a maximum 0.55 km. The alternative refit uses the C2 coordinates and keeps all 194 bed measurements and limits. The observed discrepancies provide an empirical check on registration. Both panels can share survey error. [Pixel coordinates on both maps](paired-map-coordinates/paired-coordinates.csv) · [Alternative model input](paired-map-coordinates/c2-positions.csv)

| Median-thickness volume (km³) | Published C1 positions | Alternative C2 positions |
|---|---:|---:|
| Regional GP | 3.50 | 3.50 |
| Regional cubic spline | 13.17 | 13.16 |
| Global nested corridors | 7.58 | 7.76 |

The alternative regional hull covers 279,806 km², compared with 279,806 km² for the principal fits. The alternative nested fit flags a warning in its numerical checks. [Regional fits and diagnostics](paired-map-coordinates/regional/report.json) · [Nested volume interval and diagnostics](paired-map-coordinates/result.json)

The five principal fields can also be evaluated directly at each of the two coordinates for a station. The two positions come from the original maps. This evaluation holds the fitted field fixed. Reproduce the comparison by adding `--paired-output results/paired-site-predictions.csv` to the station-prediction command above. [Fitted thickness at both original map positions](paired-map-coordinates/principal-predictions.csv)

## Marine surveys with a mapped location and no thickness limit

The three global families predict latent thickness at all 102 marine survey symbols. The table preserves the inspected or unresolved horizon classification, the site coordinates, and each model’s median and conditional 95% interval. The survey symbols supply no detection threshold. Some coincide with independently measured beds. Rebuild this table by adding `--probe-output results/pinatubo-probe-predictions.csv` to the station-prediction command. [Predictions by survey location](marine-probe-predictions.csv) · [Original surveys and source symbols](../../../../data/audits/Pinatubo-1991/marine-check.md)
