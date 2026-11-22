# Pinatubo 1991 Layer C · Spherical mixture

The principal bulk volume is **5.88 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **4.05–28.7 km³** across complete posterior draws.

![Pinatubo 1991 Layer C, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

Every contour on these maps is conditional. [Spatial uncertainty and measurement footprint](field.png)

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 5.0–37.6 |
| Unknown fan, 90° half-thickness angle | 4.1–26.6 |
| Unknown fan, 47° half-thickness angle | 3.9–20.9 |
| Enhanced narrow hidden fan, separate sensitivity | 4.2–41.9 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 2.9–25.0 | Passed |
| 45° | 4.8–35.3 | Passed |

The median fraction inside the screened measurement footprint is **10.3%**. The median fraction beyond the farthest exact site is **2.6%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **0.813**, compared with 0.826 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **97.1%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 56 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.161 for the principal field and 1.204 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 4.1–28.7 | Passed |
| diffuse tail | 4.1–53.9 | Passed |
| wide geography | 4.1–291.4 | Passed |
| volume prior shift | 4.1–25.2 | Passed |
| one fan | 0.9–1.9 | Passed |
| five fans | 4.3–703.0 | Passed |
| gaussian errors | 4.0–26.0 | parent sampler warning |
| diffuse mixture weights | 3.8–10.8 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
