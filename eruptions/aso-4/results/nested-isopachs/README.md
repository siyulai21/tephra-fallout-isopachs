# Aso-4 · Bayesian nested corridors

The principal bulk volume is **432 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **197–1.01e+04 km³** across complete posterior draws.

![Aso-4, Bayesian nested corridors. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

The buffered geographic score is 1.556 for the corridor and 1.540 for the whole-area sensitivity. Lower scores indicate better predictions at withheld sites. The corridor’s 95% predictive intervals cover 76 of 83 exact withheld readings. 3 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check passed, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Corridor 1 cm area (km²) | Whole-area 1 cm area (km²) | Corridor conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 8 | 0 | 2,237,534 | 4,859,178 | 21.1–748.6 |
| NE | 60 | 4 | 4,253,288 | 5,661,925 | 93.9–780.1 |
| E | 11 | 0 | 5,127,468 | 7,852,889 | 88.9–1,032.4 |
| SE | 2 | 0 | 1,913,533 | 10,464,919 | 17.2–1,601.3 |
| S | 2 | 0 | 442,462 | 9,962,320 | 2.9–1,796.6 |
| SW | 0 | 0 | 58,336 | 8,021,379 | 0.5–1,717.6 |
| W | 0 | 0 | 48,254 | 6,319,337 | 0.3–1,347.0 |
| NW | 0 | 0 | 244,672 | 5,163,476 | 1.9–1,007.5 |

At 1 cm, the corridor median map encloses 14,325,546 km² and the whole-area sensitivity encloses 58,305,423 km². Of these, 13,620,556 and 57,600,434 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 25.0%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

[Corridor and whole-area maps on a common scale](paired-isopachs.png)

| Prior probability of broad continuation | Conditional 95% volume interval (km³) |
|---:|---:|
| 10% | 246.1–4,976.9 |
| 25% | 249.5–9,990.1 |
| 50% | 258.8–14,616.6 |
| 75% | 278.0–15,558.3 |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.

[Whole-area sensitivity report, posterior and diagnostics](../nested-whole-area/README.md)
