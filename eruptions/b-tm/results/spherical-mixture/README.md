# B-Tm · Spherical mixture

The principal bulk volume is **37.1 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **26–58 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![B-Tm, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 63 of 71 modeled locations. The full volume also includes fallout thinner than the displayed footprint. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 30.7–340.9 |
| Unknown fan, 90° half-thickness angle | 30.1–263.9 |
| Unknown fan, 47° half-thickness angle | 29.8–212.0 |
| Enhanced narrow hidden fan, separate sensitivity | 30.4–305.2 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 27.8–217.9 | Passed |
| 45° | 30.4–302.6 | Warning |

The median fraction inside the screened measurement footprint is **12.1%**. The median fraction beyond the farthest exact site is **28.1%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.487**, compared with 1.492 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **93.8%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 3 held-out readings with gap exposure of at least 0.5, mean negative log score was 0.326 for the principal field and 0.392 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 30.2–275.5 | Passed |
| diffuse tail | 29.5–295.2 | Passed |
| wide geography | 30.4–782.9 | volume draw split warning |
| volume prior shift | 30.2–340.2 | Passed |
| one fan | 30.1–64.5 | Passed |
| five fans | 30.2–292.3 | parent sampler warning |
| gaussian errors | 25.2–188.3 | parent sampler warning |
| diffuse mixture weights | 31.2–158.6 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

Geographic folds with sampling warnings (numbered from one) include 1, 4.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
