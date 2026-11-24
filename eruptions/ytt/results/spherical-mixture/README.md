# YTT · Spherical mixture

The principal bulk volume is **1,227 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **980–1.52e+03 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![YTT, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 42 of 46 modeled locations. The full volume also includes fallout thinner than the displayed footprint. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 1,200.9–53,850.6 |
| Unknown fan, 90° half-thickness angle | 1,066.9–47,785.2 |
| Unknown fan, 47° half-thickness angle | 1,008.8–42,654.0 |
| Enhanced narrow hidden fan, separate sensitivity | 1,148.8–53,847.2 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 867.4–47,627.0 | Passed |
| 45° | 1,179.3–51,671.4 | Passed |

The median fraction inside the screened measurement footprint is **14.0%**. The median fraction beyond the farthest exact site is **0.9%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.756**, compared with 1.777 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **84.8%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 3 held-out readings with gap exposure of at least 0.5, mean negative log score was 2.147 for the principal field and 1.753 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 1,054.3–48,708.5 | Passed |
| diffuse tail | 1,062.2–43,260.0 | Passed |
| wide geography | 1,083.8–109,462.6 | Passed |
| volume prior shift | 1,069.2–96,044.0 | Passed |
| one fan | 1,013.4–1,771.3 | Passed |
| five fans | 1,071.7–45,159.1 | Passed |
| gaussian errors | 1,022.0–44,551.5 | Passed |
| diffuse mixture weights | 1,144.7–11,157.7 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 96 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
