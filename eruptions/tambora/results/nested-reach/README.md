# Tambora · Geographic reach of nested isopachs

The conditional bulk fallout estimate is **112.7 km³** (95% interval **62.5–237.7**). The median thickness map integrates to 94.9 km³. The earlier whole-area rank gives 146.9 km³ (96.1–290.7).

![Previous and revised median isopachs](isopachs.png)

Solid portions cross a retained triangle with thicknesses on either side at distinct measured sites and at least 90% posterior crossing probability. Dashed portions depend on the continuation. The triangles diagnose spatial coverage. The shared contour screen also checks thickness bracketing and deletion stability. [Whole-Earth field](world.png) · [Every local bracket](local-brackets.csv.gz) · [Complete contour checks](support.json)

The buffered geographic score is 1.646, compared with 1.368 previously. Lower scores indicate better predictions at withheld sites. 4 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check passed, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Old 1 cm area (km²) | Revised 1 cm area (km²) | Revised volume (km³), median [95%] |
|---|---:|---:|---:|---:|---:|
| N | 2 | 0 | 174,441 | 159,305 | 13.4 [6.0–29.0] |
| NE | 3 | 0 | 169,893 | 157,692 | 14.6 [5.1–33.9] |
| E | 4 | 0 | 173,887 | 132,324 | 11.7 [3.4–29.6] |
| SE | 2 | 0 | 121,441 | 46,306 | 4.3 [1.2–35.7] |
| S | 1 | 0 | 131,175 | 66,038 | 5.8 [1.6–49.7] |
| SW | 1 | 0 | 225,471 | 146,949 | 13.6 [4.7–55.3] |
| W | 13 | 0 | 305,697 | 279,086 | 24.0 [14.8–40.5] |
| NW | 0 | 0 | 262,780 | 210,411 | 16.3 [4.8–36.6] |

At 1 cm, the two median maps enclose 1,564,785 and 1,198,110 km² across Earth. Of these, 1,515,575 and 1,148,900 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 25.0%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

| Prior probability of broad continuation | Median bulk volume, km³ [conditional 95%] |
|---:|---:|
| 10% | 107.8 [61.8–211.6] |
| 25% | 112.7 [62.5–237.7] |
| 50% | 123.4 [64.2–276.0] |
| 75% | 135.3 [68.1–299.7] |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.
