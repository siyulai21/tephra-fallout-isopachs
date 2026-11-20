# Minoan spherical-mixture fallout

**40.8 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **24.6–1,989.1 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 46.0 [28.1–2,529.2] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 45.6 [28.2–2,353.3] |
| Unknown fan, 90° half-thickness angle | 39.5 [24.9–1,841.1] |
| Unknown fan, 47° half-thickness angle | 36.4 [23.4–1,589.0] |
| Enhanced narrow hidden fan, separate sensitivity | 45.5 [27.0–2,424.7] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 38.4 [21.2–1,990.9] | Warning |
| 45° | 44.2 [26.9–2,280.9] | Warning |

The median fraction inside the screened measurement footprint is **6.0%**. The median fraction beyond the farthest exact site is **11.6%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.575**, compared with 1.570 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **94.6%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 38 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.559 for the principal field and 1.509 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 40.8 [24.6–1,989.1] | Passed |
| diffuse tail | 51.8 [25.6–3,345.1] | Passed |
| wide geography | 59.7 [26.1–19,068.0] | volume draw split warning |
| volume prior shift | 44.8 [24.8–4,283.2] | Passed |
| one fan | 32.0 [23.1–47.3] | Passed |
| five fans | 47.3 [25.0–2,885.9] | Passed |
| gaussian errors | 41.6 [24.7–2,430.8] | Passed |
| diffuse mixture weights | 56.7 [28.6–417.5] | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
