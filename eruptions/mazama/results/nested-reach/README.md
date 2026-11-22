# Mazama · Geographic reach of nested isopachs

The conditional bulk fallout estimate is **271.4 km³** (95% interval **129.2–938.0**). The median thickness map integrates to 244.6 km³. The earlier whole-area rank gives 411.2 km³ (162.2–1,663.6).

![Previous and revised median isopachs](isopachs.png)

Solid portions cross a retained triangle with thicknesses on either side at distinct measured sites and at least 90% posterior crossing probability. Dashed portions depend on the continuation. The triangles diagnose spatial coverage. The shared contour screen also checks thickness bracketing and deletion stability. [Whole-Earth field](world.png) · [Every local bracket](local-brackets.csv.gz) · [Complete contour checks](support.json)

The buffered geographic score is 1.365, compared with 1.503 previously. Lower scores indicate better predictions at withheld sites. 4 of 4 geographic fits passed their sampling checks. The full-data proposal and importance check passed, the final volume populations passed, and the integration check passed. The shared screen accepted a complete contour band under 0 policies. [Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)

## Where the volume lies

![Full compass, measured values, median contours and conditional uncertainty](compass.png)

The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.

| Compass sector | Exact sites | Bounded sites | Old 1 cm area (km²) | Revised 1 cm area (km²) | Revised volume (km³), median [95%] |
|---|---:|---:|---:|---:|---:|
| N | 21 | 0 | 416,621 | 387,064 | 30.2 [21.9–94.8] |
| NE | 19 | 0 | 1,366,297 | 970,609 | 103.1 [60.7–252.1] |
| E | 3 | 0 | 1,161,925 | 693,944 | 100.3 [33.2–333.3] |
| SE | 0 | 0 | 407,034 | 173,801 | 24.7 [1.4–177.3] |
| S | 0 | 0 | 94,921 | 13,044 | 1.8 [0.1–56.0] |
| SW | 0 | 0 | 28,185 | 5,771 | 1.0 [0.1–20.7] |
| W | 0 | 0 | 20,188 | 10,289 | 1.3 [0.3–16.8] |
| NW | 2 | 0 | 35,920 | 28,999 | 3.2 [1.7–28.5] |

At 1 cm, the two median maps enclose 3,531,090 and 2,283,521 km² across Earth. Of these, 3,357,213 and 2,109,644 km² fall outside the retained exact-site triangles. An unsampled sector can contain ash. Its allocation depends on the assumed continuation.

The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. The fitted probability of whole-bearing continuation is 25.0%. The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. Only positive evidence supplies a potential dispersal corridor.

[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)

## Structural uncertainty

| Prior probability of broad continuation | Median bulk volume, km³ [conditional 95%] |
|---:|---:|
| 10% | 256.4 [126.8–720.4] |
| 25% | 271.4 [129.2–938.0] |
| 50% | 304.9 [139.0–1,264.4] |
| 75% | 356.5 [149.0–1,465.2] |

The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it.
