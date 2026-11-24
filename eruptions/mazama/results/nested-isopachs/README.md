# Mazama · Bayesian nested corridors

The principal bulk volume is **175 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **106–605 km³** across complete posterior draws.

![Mazama, Bayesian nested corridors. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

The buffered geographic score is 0.384 for the corridor and 0.334 for the whole-area sensitivity. Lower scores indicate better predictions at withheld sites. The corridor’s 95% predictive intervals cover 27 of 31 exact withheld readings. 3 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check passed, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Corridor 1 cm area (km²) | Whole-area 1 cm area (km²) | Corridor conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 13 | 89 | 237,463 | 251,812 | 22.7–114.2 |
| NE | 3 | 89 | 299,997 | 365,422 | 33.3–149.2 |
| E | 11 | 36 | 267,661 | 270,350 | 24.7–118.0 |
| SE | 3 | 22 | 162,817 | 205,048 | 11.8–100.3 |
| S | 0 | 1 | 14,651 | 93,014 | 1.1–52.8 |
| SW | 0 | 0 | 299 | 27,196 | 0.097–15.3 |
| W | 0 | 1 | 1,126 | 17,900 | 0.5–21.2 |
| NW | 1 | 2 | 39,981 | 75,210 | 4.6–63.1 |

At 1 cm, the corridor median map encloses 1,023,995 km² and the whole-area sensitivity encloses 1,305,950 km². Of these, 930,673 and 1,213,667 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 7.5%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

[Corridor and whole-area maps on a common scale](paired-isopachs.png)

| Prior probability of broad continuation | Conditional 95% volume interval (km³) |
|---:|---:|
| 10% | 113.4–561.3 |
| 25% | 114.3–569.6 |
| 50% | 115.6–610.2 |
| 75% | 120.3–678.9 |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.

[Whole-area sensitivity report, posterior and diagnostics](../nested-whole-area/README.md)
