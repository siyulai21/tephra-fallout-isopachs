# Pinatubo 1991 Layer C · Bayesian nested corridors

The principal bulk volume is **2.12 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **1.46–6.93 km³** across complete posterior draws.

![Pinatubo 1991 Layer C, Bayesian nested corridors. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

Solid segments cross locally bracketed triangles with at least 90% posterior crossing probability. [Whole-Earth context](world.png) · [Local brackets](local-brackets.csv.gz) · [Complete contour checks](support.json)

The buffered geographic score is 1.440 for the corridor and 1.076 for the whole-area sensitivity. Lower scores indicate better predictions at withheld sites. The corridor’s 95% predictive intervals cover 84 of 102 exact withheld readings. 2 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check passed, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Corridor 1 cm area (km²) | Whole-area 1 cm area (km²) | Corridor conditional 95% volume interval (km³) |
|---|---:|---:|---:|---:|---:|
| N | 1 | 2 | 367 | 1,056 | 0.023–0.3 |
| NE | 8 | 2 | 522 | 606 | 0.032–0.2 |
| E | 28 | 8 | 830 | 905 | 0.047–0.3 |
| SE | 17 | 7 | 1,441 | 2,002 | 0.088–0.8 |
| S | 7 | 6 | 1,680 | 3,589 | 0.1–1.3 |
| SW | 20 | 4 | 3,204 | 4,510 | 0.5–1.8 |
| W | 14 | 4 | 3,347 | 3,911 | 0.5–1.6 |
| NW | 7 | 2 | 966 | 2,333 | 0.077–0.8 |

At 1 cm, the corridor median map encloses 12,357 km² and the whole-area sensitivity encloses 18,911 km². Of these, 11,449 and 18,004 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 9.6%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

[Corridor and whole-area maps on a common scale](paired-isopachs.png)

| Prior probability of broad continuation | Conditional 95% volume interval (km³) |
|---:|---:|
| 10% | 1.5–5.7 |
| 25% | 1.5–6.9 |
| 50% | 1.5–8.1 |
| 75% | 1.5–10.2 |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.

[Whole-area sensitivity report, posterior and diagnostics](../nested-whole-area/README.md)
