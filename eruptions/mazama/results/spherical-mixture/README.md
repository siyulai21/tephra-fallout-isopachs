# Mazama · Spherical mixture

**Historical comparison on 271 localities.** The eruption catalogue uses the latest audited input.

The principal bulk volume is **241 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **165–403 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Mazama, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 229 of 271 modeled locations. The thickest displayed line is 4096 cm and encloses 5 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 164.2–1,539.3 |
| Unknown fan, 90° half-thickness angle | 160.5–1,512.3 |
| Unknown fan, 47° half-thickness angle | 159.1–1,496.2 |
| Enhanced narrow hidden fan, separate sensitivity | 163.7–1,522.6 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 154.8–1,545.8 | Passed |
| 45° | 162.0–1,620.9 | Passed |

The median fraction inside the screened measurement footprint is **25.0%**. The median fraction beyond the farthest exact site is **1.2%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **0.494**, compared with 0.476 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **71.0%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 45 held-out readings with gap exposure of at least 0.5, mean negative log score was 0.820 for the principal field and 0.738 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 160.9–1,523.2 | Passed |
| diffuse tail | 160.6–1,566.0 | Passed |
| wide geography | 163.3–2,504.6 | Passed |
| volume prior shift | 160.7–2,025.8 | Passed |
| one fan | 305.3–95,039.0 | volume draw split warning |
| five fans | 162.5–5,211.2 | volume draw split warning |
| gaussian errors | 153.8–757.6 | Passed |
| diffuse mixture weights | 154.0–584.4 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 96 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)

[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)
