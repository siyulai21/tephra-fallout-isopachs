# Ilopango TBJ (Unit G) spherical-mixture fallout

**190.9 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **62.9–3,300.8 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 255.6 [78.3–4,482.4] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 255.6 [78.3–4,482.4] |
| Unknown fan, 90° half-thickness angle | 182.9 [63.6–3,096.7] |
| Unknown fan, 47° half-thickness angle | 151.6 [57.3–2,544.9] |
| Enhanced narrow hidden fan, separate sensitivity | 220.1 [68.3–4,160.2] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 155.7 [47.5–2,973.5] | Warning |
| 45° | 226.2 [73.7–4,266.0] | Passed |

The median fraction inside the screened measurement footprint is **0.8%**. The median fraction beyond the farthest exact site is **61.3%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.225**, compared with 1.173 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **94.2%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 18 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.361 for the principal field and 1.192 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 190.9 [62.9–3,300.8] | volume draw split warning |
| diffuse tail | 276.7 [79.1–4,336.7] | Passed |
| wide geography | 350.4 [98.0–19,135.6] | Passed |
| volume prior shift | 210.7 [64.0–5,448.9] | volume draw split warning |
| one fan | 64.7 [43.3–108.3] | Passed |
| five fans | 219.6 [64.8–4,110.7] | Passed |
| gaussian errors | 209.7 [43.3–21,116.2] | Passed |
| diffuse mixture weights | 235.1 [66.4–1,899.2] | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 384 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
