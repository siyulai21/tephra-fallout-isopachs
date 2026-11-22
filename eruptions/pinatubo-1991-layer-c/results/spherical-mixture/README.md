# Pinatubo 1991 (Layer C) · Spherical mixture

The principal bulk volume is **13.7 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **8.79–25.4 km³** across complete posterior draws. **Numerical warning on the fitted deposit distribution.**

![Pinatubo 1991 (Layer C), Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

Every contour on these maps is conditional. [Spatial uncertainty and measurement footprint](field.png)

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 9.6–27.1 |
| Unknown fan, 90° half-thickness angle | 8.7–24.4 |
| Unknown fan, 47° half-thickness angle | 8.4–22.4 |
| Enhanced narrow hidden fan, separate sensitivity | 9.2–32.8 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 7.5–24.9 | Passed |
| 45° | 9.8–27.4 | Passed |

The median fraction inside the screened measurement footprint is **4.1%**. The median fraction beyond the farthest exact site is **1.1%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **0.986**, compared with 1.020 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **97.1%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 79 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.387 for the principal field and 1.466 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 8.8–25.4 | parent sampler warning |
| diffuse tail | 10.9–155.2 | Passed |
| wide geography | 9.3–26.2 | Passed |
| volume prior shift | 7.5–25.5 | parent sampler warning |
| one fan | 14.6–34.8 | Passed |
| five fans | 6.5–431.0 | volume draw split warning |
| gaussian errors | 6.3–10.1 | Passed |
| diffuse mixture weights | 8.4–22.9 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 384 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

Geographic folds with sampling warnings (numbered from one) include 2, 3.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)

[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)
