# Sakurajima Satsuma (P14) · Spherical mixture

The principal bulk volume is **6.94 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **4.22–14.5 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Sakurajima Satsuma (P14), Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 4 cm footprint includes 55 of 57 modeled locations. The thickest displayed line is 32 cm and encloses 11 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 6.4–837.5 |
| Unknown fan, 90° half-thickness angle | 5.3–604.6 |
| Unknown fan, 47° half-thickness angle | 4.8–489.6 |
| Enhanced narrow hidden fan, separate sensitivity | 5.8–948.7 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 3.7–536.0 | Passed |
| 45° | 6.1–753.5 | Passed |

The median fraction inside the screened measurement footprint is **1.6%**. The median fraction beyond the farthest exact site is **49.8%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.376**, compared with 1.513 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **88.0%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 23 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.842 for the principal field and 2.210 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 5.1–666.5 | Passed |
| diffuse tail | 5.1–1,018.6 | Passed |
| wide geography | 5.7–3,337.1 | Passed |
| volume prior shift | 5.1–1,674.1 | volume draw split warning |
| one fan | 4.5–11.2 | Passed |
| five fans | 5.3–1,151.9 | Passed |
| gaussian errors | 5.1–499.5 | Passed |
| diffuse mixture weights | 5.4–69.9 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 96 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
