# Tambora · Spherical mixture

The principal bulk volume is **117 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **82.3–176 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Tambora, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 21 of 26 modeled locations. The full volume also includes fallout thinner than the displayed footprint. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 113.5–1,224.9 |
| Unknown fan, 90° half-thickness angle | 101.1–817.4 |
| Unknown fan, 47° half-thickness angle | 96.1–698.5 |
| Enhanced narrow hidden fan, separate sensitivity | 105.7–1,125.6 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 83.8–837.3 | Passed |
| 45° | 110.2–1,072.8 | Passed |

The median fraction inside the screened measurement footprint is **7.1%**. The median fraction beyond the farthest exact site is **5.4%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.477**, compared with 1.471 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **100.0%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 11 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.702 for the principal field and 1.685 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 100.8–920.1 | Passed |
| diffuse tail | 99.8–1,544.0 | Passed |
| wide geography | 106.9–3,846.6 | volume draw split warning |
| volume prior shift | 99.0–1,112.6 | Passed |
| one fan | 97.3–707.6 | Passed |
| five fans | 99.6–1,007.8 | volume draw split warning |
| gaussian errors | 93.0–713.4 | Passed |
| diffuse mixture weights | 108.1–824.2 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
