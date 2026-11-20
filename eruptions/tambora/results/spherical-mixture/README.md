# Tambora spherical-mixture fallout

**162.3 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **100.8–920.1 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 179.1 [112.9–1,284.2] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 178.7 [113.5–1,224.9] |
| Unknown fan, 90° half-thickness angle | 158.8 [101.1–817.4] |
| Unknown fan, 47° half-thickness angle | 149.6 [96.1–698.5] |
| Enhanced narrow hidden fan, separate sensitivity | 176.8 [105.7–1,125.6] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 147.8 [83.8–837.3] | Passed |
| 45° | 173.4 [110.2–1,072.8] | Passed |

The median fraction inside the screened measurement footprint is **7.1%**. The median fraction beyond the farthest exact site is **5.4%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.477**, compared with 1.471 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **100.0%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 11 held-out readings with gap exposure of at least 0.5, mean negative log score was 1.702 for the principal field and 1.685 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 162.3 [100.8–920.1] | Passed |
| diffuse tail | 169.3 [99.8–1,544.0] | Passed |
| wide geography | 214.2 [106.9–3,846.6] | volume draw split warning |
| volume prior shift | 163.6 [99.0–1,112.6] | Passed |
| one fan | 152.0 [97.3–707.6] | Passed |
| five fans | 162.8 [99.6–1,007.8] | volume draw split warning |
| gaussian errors | 149.4 [93.0–713.4] | Passed |
| diffuse mixture weights | 198.2 [108.1–824.2] | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 192 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
