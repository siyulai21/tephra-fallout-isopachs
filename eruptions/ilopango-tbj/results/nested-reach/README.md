# Ilopango TBJ (Unit G) · Geographic reach of nested isopachs

The conditional bulk fallout estimate is **40.9 km³** (95% interval **25.1–94.2**). The median thickness map integrates to 39.1 km³. The earlier whole-area rank gives 50.5 km³ (37.5–181.8).

![Previous and revised median isopachs](isopachs.png)

Solid portions cross a retained triangle with thicknesses on either side at distinct measured sites and at least 90% posterior crossing probability. Dashed portions depend on the continuation. The triangles diagnose spatial coverage. The shared contour screen also checks thickness bracketing and deletion stability. [Whole-Earth field](world.png) · [Every local bracket](local-brackets.csv.gz) · [Complete contour checks](support.json)

The buffered geographic score is 1.274, compared with 1.159 previously. Lower scores indicate better predictions at withheld sites. 1 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check retained a warning, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Old 1 cm area (km²) | Revised 1 cm area (km²) | Revised volume (km³), median [95%] |
|---|---:|---:|---:|---:|---:|
| N | 3 | 0 | 65,640 | 43,953 | 4.4 [2.1–12.4] |
| NE | 2 | 1 | 65,728 | 45,138 | 4.6 [2.5–12.2] |
| E | 9 | 1 | 65,829 | 44,327 | 4.5 [2.2–13.6] |
| SE | 5 | 1 | 65,828 | 56,550 | 5.4 [3.5–13.0] |
| S | 12 | 0 | 65,717 | 59,081 | 5.7 [4.3–11.5] |
| SW | 2 | 0 | 65,036 | 57,776 | 5.4 [3.2–10.3] |
| W | 12 | 1 | 64,541 | 60,289 | 5.5 [3.3–10.1] |
| NW | 7 | 0 | 65,529 | 54,117 | 5.1 [2.7–10.8] |

At 1 cm, the two median maps enclose 523,848 and 421,230 km² across Earth. Of these, 508,195 and 405,577 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 25.0%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

| Prior probability of broad continuation | Median bulk volume, km³ [conditional 95%] |
|---:|---:|
| 10% | 39.0 [24.8–69.8] |
| 25% | 40.9 [25.1–94.2] |
| 50% | 44.4 [25.9–125.6] |
| 75% | 47.6 [27.6–147.1] |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.
