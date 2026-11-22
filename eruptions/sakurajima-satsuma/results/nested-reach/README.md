# Sakurajima Satsuma (P14) · Geographic reach of nested isopachs

The conditional bulk fallout estimate is **6.7 km³** (95% interval **3.1–30.7**). The median thickness map integrates to 6.3 km³. The earlier whole-area rank gives 12.5 km³ (7.0–60.2).

![Previous and revised median isopachs](isopachs.png)

Solid portions cross a retained triangle with thicknesses on either side at distinct measured sites and at least 90% posterior crossing probability. Dashed portions depend on the continuation. The triangles diagnose spatial coverage. The shared contour screen also checks thickness bracketing and deletion stability. [Whole-Earth field](world.png) · [Every local bracket](local-brackets.csv.gz) · [Complete contour checks](support.json)

The buffered geographic score is 1.016, compared with 0.742 previously. Lower scores indicate better predictions at withheld sites. 1 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check retained a warning, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Old 1 cm area (km²) | Revised 1 cm area (km²) | Revised volume (km³), median [95%] |
|---|---:|---:|---:|---:|---:|
| N | 2 | 2 | 15,452 | 8,378 | 0.7 [0.3–3.2] |
| NE | 6 | 1 | 4,458 | 3,562 | 0.2 [0.1–1.0] |
| E | 20 | 1 | 9,266 | 5,309 | 0.4 [0.3–2.0] |
| SE | 8 | 0 | 28,682 | 13,720 | 1.2 [0.5–5.8] |
| S | 6 | 2 | 35,244 | 15,942 | 1.4 [0.8–6.7] |
| SW | 3 | 1 | 17,479 | 9,452 | 0.7 [0.3–3.7] |
| W | 3 | 0 | 17,554 | 13,992 | 0.9 [0.6–3.7] |
| NW | 2 | 0 | 25,815 | 15,021 | 1.1 [0.1–5.0] |

At 1 cm, the two median maps enclose 153,951 and 85,376 km² across Earth. Of these, 153,314 and 84,739 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 24.9%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

| Prior probability of broad continuation | Median bulk volume, km³ [conditional 95%] |
|---:|---:|
| 10% | 6.1 [3.1–18.9] |
| 25% | 6.7 [3.1–30.7] |
| 50% | 8.3 [3.2–43.9] |
| 75% | 10.7 [3.4–50.5] |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.
