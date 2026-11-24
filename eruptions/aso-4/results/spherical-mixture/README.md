# Aso-4 · Spherical mixture

The principal bulk volume is **473 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **282–942 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Aso-4, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 2 cm footprint includes 84 of 87 modeled locations. The full volume also includes fallout thinner than the displayed footprint. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 508.5–40,136.3 |
| Unknown fan, 90° half-thickness angle | 413.7–23,647.8 |
| Unknown fan, 47° half-thickness angle | 374.1–16,525.6 |
| Enhanced narrow hidden fan, separate sensitivity | 457.6–42,645.3 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 314.4–26,030.3 | Passed |
| 45° | 470.0–34,457.4 | Warning |

The median fraction inside the screened measurement footprint is **2.8%**. The median fraction beyond the farthest exact site is **10.6%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.632**, compared with 1.632 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **95.2%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 13 held-out readings with gap exposure of at least 0.5, mean negative log score was 2.045 for the principal field and 2.088 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 402.9–27,836.6 | Passed |
| diffuse tail | 396.0–25,212.1 | volume draw split warning |
| wide geography | 543.6–147,821.0 | Passed |
| volume prior shift | 419.6–49,587.6 | Passed |
| one fan | 2,176.2–85,084,664.8 | volume draw split warning |
| five fans | 411.4–28,624.8 | volume draw split warning |
| gaussian errors | 402.6–25,865.4 | volume draw split warning |
| diffuse mixture weights | 379.9–4,928.9 | Passed |

At 384 parent draws per chain, the following resolved contour areas still differed by at least 10% between draw halves: S 10 cm (10%), SW 10 cm (18%).

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
