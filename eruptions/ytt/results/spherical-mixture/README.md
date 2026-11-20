# YTT spherical-mixture fallout

**1,510.5 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **1,054.3–48,708.5 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 1,674.2 [1,203.6–54,906.9] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 1,672.8 [1,200.9–53,850.6] |
| Unknown fan, 90° half-thickness angle | 1,463.9 [1,066.9–47,785.2] |
| Unknown fan, 47° half-thickness angle | 1,368.9 [1,008.8–42,654.0] |
| Enhanced narrow hidden fan, separate sensitivity | 1,660.6 [1,148.8–53,847.2] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 1,385.8 [867.4–47,627.0] | Passed |
| 45° | 1,635.8 [1,179.3–51,671.4] | Passed |

The median fraction inside the screened measurement footprint is **14.0%**. The median fraction beyond the farthest exact site is **0.9%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.756**, compared with 1.777 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **84.8%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 3 held-out readings with gap exposure of at least 0.5, mean negative log score was 2.147 for the principal field and 1.753 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 1,510.5 [1,054.3–48,708.5] | Passed |
| diffuse tail | 1,480.1 [1,062.2–43,260.0] | Passed |
| wide geography | 1,579.1 [1,083.8–109,462.6] | Passed |
| volume prior shift | 1,560.5 [1,069.2–96,044.0] | Passed |
| one fan | 1,312.5 [1,013.4–1,771.3] | Passed |
| five fans | 1,553.9 [1,071.7–45,159.1] | Passed |
| gaussian errors | 1,443.8 [1,022.0–44,551.5] | Passed |
| diffuse mixture weights | 2,114.0 [1,144.7–11,157.7] | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 96 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
