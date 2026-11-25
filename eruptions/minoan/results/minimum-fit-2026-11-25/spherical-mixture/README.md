# Minoan · Spherical mixture

The principal bulk volume is **26.5 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **20.2–38.5 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Minoan, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 91 of 101 modeled locations. The thickest displayed line is 16 cm and encloses 15 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 26.1–1,322.6 |
| Unknown fan, 90° half-thickness angle | 22.9–831.9 |
| Unknown fan, 47° half-thickness angle | 21.5–630.7 |
| Enhanced narrow hidden fan, separate sensitivity | 24.8–1,230.5 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 19.4–845.8 | Passed |
| 45° | 25.0–1,098.4 | Passed |

The median fraction inside the screened measurement footprint is **7.1%**. The median fraction beyond the farthest exact site is **16.1%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.569**, compared with 1.562 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **95.0%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 38 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.514 for the principal field and 1.455 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 22.6–925.3 | Passed |
| diffuse tail | 23.6–1,277.3 | Passed |
| wide geography | 22.9–2,622.4 | volume draw split warning |
| volume prior shift | 22.8–1,317.7 | volume draw split warning |
| one fan | 21.6–41.7 | Passed |
| five fans | 22.9–1,021.4 | Passed |
| gaussian errors | 22.8–861.7 | Passed |
| diffuse mixture weights | 26.1–366.0 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
