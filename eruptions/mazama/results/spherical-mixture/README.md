# Mazama · Spherical mixture

The principal bulk volume is **370 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **212–7.17e+03 km³** across complete posterior draws.

![Mazama, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 227.4–7,612.4 |
| Unknown fan, 90° half-thickness angle | 220.7–7,518.0 |
| Unknown fan, 47° half-thickness angle | 217.8–7,308.6 |
| Enhanced narrow hidden fan, separate sensitivity | 225.0–7,682.9 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 215.2–7,441.6 | Passed |
| 45° | 225.8–7,602.0 | Passed |

The median fraction inside the screened measurement footprint is **8.4%**. The median fraction beyond the farthest exact site is **1.3%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **0.923**, compared with 0.903 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **63.8%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 63 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.957 for the principal field and 1.893 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 221.6–7,502.9 | Passed |
| diffuse tail | 220.7–7,677.3 | Passed |
| wide geography | 240.7–56,712.0 | Passed |
| volume prior shift | 224.7–14,746.4 | volume draw split warning |
| one fan | 568.8–199,544.9 | Passed |
| five fans | 221.6–9,199.2 | Passed |
| gaussian errors | 237.2–7,454.1 | Passed |
| diffuse mixture weights | 209.2–1,025.7 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 96 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)

[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)
