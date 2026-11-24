# Mazama · Spherical mixture

The principal bulk volume is **418 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **248–795 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth. **Numerical warning on the fitted deposit distribution.**

![Mazama, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 280 of 300 modeled locations. The thickest displayed line is 8192 cm and encloses 6 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 227.3–5,891.2 |
| Unknown fan, 90° half-thickness angle | 221.7–5,423.9 |
| Unknown fan, 47° half-thickness angle | 220.1–5,232.7 |
| Enhanced narrow hidden fan, separate sensitivity | 225.6–5,597.3 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 218.3–5,265.2 | Passed |
| 45° | 226.4–5,736.8 | Passed |

The median fraction inside the screened measurement footprint is **4.2%**. The median fraction beyond the farthest exact site is **0.8%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.050**, compared with 1.065 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **60.9%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 30 held-out readings with gap exposure of at least 0.5, mean negative log score was 2.806 for the principal field and 3.023 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 223.1–5,503.7 | parent sampler warning |
| diffuse tail | 248.0–5,670.9 | parent sampler warning |
| wide geography | 235.8–13,812.5 | parent sampler warning |
| volume prior shift | 226.2–8,126.0 | parent sampler, volume draw split warning |
| one fan | 506.4–320,705.3 | parent sampler warning |
| five fans | 205.2–4,619.8 | Passed |
| gaussian errors | 226.5–7,088.7 | Passed |
| diffuse mixture weights | 201.1–1,333.0 | parent sampler warning |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)

[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)
