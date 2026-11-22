# Mazama · Spherical mixture

The principal bulk volume is **171 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **135–914 km³** across complete posterior draws.

![Mazama, Spherical mixture. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## What the unobserved directions contribute

| Continuation assumption | Conditional 95% volume interval (km³) |
|---|---:|
| Broad parent continuation | 145.3–3,628.7 |
| Unknown fan, 90° half-thickness angle | 139.6–2,514.1 |
| Unknown fan, 47° half-thickness angle | 137.4–1,849.3 |
| Enhanced narrow hidden fan, separate sensitivity | 143.8–3,181.3 |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| 11.25° | 132.1–2,511.4 | Passed |
| 45° | 144.0–2,957.4 | Passed |

The median fraction inside the screened measurement footprint is **15.5%**. The median fraction beyond the farthest exact site is **0.2%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.561**, compared with 1.577 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **88.9%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 3 held-out readings with gap exposure of at least 0.5, mean negative log score was 2.315 for the principal field and 2.900 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Conditional 95% volume interval (km³) | Numerical checks |
|---|---:|---|
| baseline | 140.1–2,614.3 | Passed |
| diffuse tail | 138.8–2,819.2 | Passed |
| wide geography | 138.1–24,588.4 | parent sampler, volume draw split warning |
| volume prior shift | 137.0–5,742.4 | Passed |
| one fan | 208.9–92,709.0 | volume draw split warning |
| five fans | 141.5–3,537.7 | Passed |
| gaussian errors | 152.3–1,320.5 | parent sampler warning |
| diffuse mixture weights | 137.7–592.0 | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 384 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

Geographic folds with sampling warnings (numbered from one) include 3.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
