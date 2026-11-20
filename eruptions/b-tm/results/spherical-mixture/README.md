# B-Tm spherical-mixture fallout

**47.0 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **30.2–275.5 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 49.1 [30.7–341.7] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 48.7 [30.7–340.9] |
| Unknown fan, 90° half-thickness angle | 46.7 [30.1–263.9] |
| Unknown fan, 47° half-thickness angle | 45.7 [29.8–212.0] |
| Enhanced narrow hidden fan, separate sensitivity | 48.1 [30.4–305.2] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 43.3 [27.8–217.9] | Passed |
| 45° | 49.2 [30.4–302.6] | Warning |

The median fraction inside the screened measurement footprint is **12.1%**. The median fraction beyond the farthest exact site is **28.1%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.487**, compared with 1.492 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **93.8%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 3 held-out readings with gap exposure of at least 0.5, mean negative log score was 0.326 for the principal field and 0.392 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 47.0 [30.2–275.5] | Passed |
| diffuse tail | 47.3 [29.5–295.2] | Passed |
| wide geography | 51.5 [30.4–782.9] | volume draw split warning |
| volume prior shift | 47.9 [30.2–340.2] | Passed |
| one fan | 47.6 [30.1–64.5] | Passed |
| five fans | 48.4 [30.2–292.3] | parent sampler warning |
| gaussian errors | 37.7 [25.2–188.3] | parent sampler warning |
| diffuse mixture weights | 49.3 [31.2–158.6] | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

Geographic folds with sampling warnings (numbered from one) include 1, 4.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)

[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)
