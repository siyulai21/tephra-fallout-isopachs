# Pinatubo 1991 (Layer C) · Spherical mixture

The principal bulk volume is **6.04 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **5.04–7.28 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Pinatubo 1991 (Layer C), Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 213 of 235 modeled locations. The thickest displayed line is 16 cm and encloses 35 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 6.3–11.0 |
| Unknown fan, 90° half-thickness angle | 5.9–9.5 |
| Unknown fan, 47° half-thickness angle | 5.7–8.9 |
| Enhanced narrow hidden fan, separate sensitivity | 6.0–13.4 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 5.0–10.2 | Passed |
| 45° | 6.3–10.5 | Passed |

The median fraction inside the screened measurement footprint is **10.8%**. The median fraction beyond the farthest exact site is **1.2%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **0.927**, compared with 0.956 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **96.6%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 76 held-out readings with gap exposure of at least 0.5, mean negative log score was 0.835 for the principal field and 0.922 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 5.9–10.2 | Passed |
| diffuse tail | 5.9–44.2 | Passed |
| wide geography | 5.5–5,935.1 | parent sampler, volume draw split warning |
| volume prior shift | 5.9–10.2 | Passed |
| one fan | 10.8–24.5 | Passed |
| five fans | 5.5–139.8 | parent sampler warning |
| gaussian errors | 5.0–7.3 | Passed |
| diffuse mixture weights | 5.3–9.7 | parent sampler warning |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 96 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

Geographic folds with sampling warnings (numbered from one) include 3.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)

[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)
