# Mazama spherical-mixture fallout

**212.9 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **140.1–2,614.3 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 223.2 [144.7–3,668.8] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 223.3 [145.3–3,628.7] |
| Unknown fan, 90° half-thickness angle | 209.9 [139.6–2,514.1] |
| Unknown fan, 47° half-thickness angle | 204.5 [137.4–1,849.3] |
| Enhanced narrow hidden fan, separate sensitivity | 219.9 [143.8–3,181.3] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 204.1 [132.1–2,511.4] | Passed |
| 45° | 219.3 [144.0–2,957.4] | Passed |

The median fraction inside the screened measurement footprint is **15.5%**. The median fraction beyond the farthest exact site is **0.2%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.561**, compared with 1.577 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **88.9%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 3 held-out readings with gap exposure of at least 0.5, mean negative log score was 2.315 for the principal field and 2.900 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 212.9 [140.1–2,614.3] | Passed |
| diffuse tail | 212.1 [138.8–2,819.2] | Passed |
| wide geography | 260.4 [138.1–24,588.4] | parent sampler, volume draw split warning |
| volume prior shift | 216.1 [137.0–5,742.4] | Passed |
| one fan | 1,025.3 [208.9–92,709.0] | volume draw split warning |
| five fans | 224.7 [141.5–3,537.7] | Passed |
| gaussian errors | 222.9 [152.3–1,320.5] | parent sampler warning |
| diffuse mixture weights | 198.0 [137.7–592.0] | Passed |

The resolved 1, 10 and 100 cm contour areas passed the split-draw test at 384 parent draws per chain. Sparsely represented or empty contour unions remain unresolved.

Geographic folds with sampling warnings (numbered from one) include 3.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
