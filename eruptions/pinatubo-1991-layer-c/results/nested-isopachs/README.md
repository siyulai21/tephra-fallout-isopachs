# Pinatubo 1991 (Layer C) · Bayesian nested corridors

The principal bulk volume is **7.75 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **7.01–9.35 km³** across complete posterior draws.

![Pinatubo 1991 (Layer C), Bayesian nested corridors. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

Solid segments cross locally bracketed triangles with at least 90% posterior crossing probability. [Whole-Earth context](world.png) · [Local brackets](local-brackets.csv.gz) · [Complete contour checks](support.json)

The buffered geographic score is 2.908 for the corridor and 2.020 for the whole-area sensitivity. Lower scores indicate better predictions at withheld sites. The corridor’s 95% predictive intervals cover 84 of 139 exact withheld readings. 1 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check passed, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Corridor 1 cm area (km²) | Whole-area 1 cm area (km²) | Corridor conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 1 | 2 | 409 | 2,386 | 0.02–0.049 |
| NE | 12 | 2 | 728 | 904 | 0.032–0.043 |
| E | 30 | 9 | 800 | 899 | 0.033–0.044 |
| SE | 19 | 10 | 1,097 | 1,552 | 0.047–0.075 |
| S | 10 | 7 | 5,023 | 15,250 | 0.2–0.4 |
| SW | 32 | 8 | 57,382 | 78,745 | 2.3–3.3 |
| W | 24 | 12 | 86,229 | 90,007 | 3.4–5.0 |
| NW | 11 | 4 | 14,617 | 24,769 | 0.5–1.0 |

At 1 cm, the corridor median map encloses 166,284 km² and the whole-area sensitivity encloses 214,513 km². Of these, 155,535 and 203,764 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 0.2%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

[Corridor and whole-area maps on a common scale](paired-isopachs.png)

| Prior probability of broad continuation | Conditional 95% volume interval (km³) |
|---:|---:|
| 10% | 7.0–9.3 |
| 25% | 7.0–9.4 |
| 50% | 7.0–9.4 |
| 75% | 7.0–9.5 |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.

[Whole-area sensitivity report, posterior and diagnostics](../nested-whole-area/README.md)
