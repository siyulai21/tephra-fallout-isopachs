# Sakurajima Satsuma (P14) spherical-mixture fallout

**11.3 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **5.1–666.5 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 14.3 [6.3–921.6] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 14.5 [6.4–837.5] |
| Unknown fan, 90° half-thickness angle | 10.8 [5.3–604.6] |
| Unknown fan, 47° half-thickness angle | 9.3 [4.8–489.6] |
| Enhanced narrow hidden fan, separate sensitivity | 13.5 [5.8–948.7] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 9.7 [3.7–536.0] | Passed |
| 45° | 13.2 [6.1–753.5] | Passed |

The median fraction inside the screened measurement footprint is **1.6%**. The median fraction beyond the farthest exact site is **49.8%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.376**, compared with 1.513 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **88.0%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 23 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.842 for the principal field and 2.210 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 11.3 [5.1–666.5] | Passed |
| diffuse tail | 15.1 [5.1–1,018.6] | Passed |
| wide geography | 15.4 [5.7–3,337.1] | Passed |
| volume prior shift | 13.2 [5.1–1,674.1] | volume draw split warning |
| one fan | 6.5 [4.5–11.2] | Passed |
| five fans | 13.3 [5.3–1,151.9] | Passed |
| gaussian errors | 11.1 [5.1–499.5] | Passed |
| diffuse mixture weights | 11.2 [5.4–69.9] | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 96 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)

[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)
