# Aso-4 spherical-mixture fallout

**1,069.7 km³ of bulk fallout** is the principal whole-Earth volume. The conditional 95% interval is **402.9–27,836.6 km³**. Each simulated deposit carries one fan direction through the unsampled ground. The map shows median thickness at each location. The volume and its interval summarize the integrals of complete deposits.

![Principal spherical mixture with uncertain continuation across unsampled bearings](field.png)

The underlying spherical mixture gave 1,466.0 [501.0–44,172.9] km³. The extension leaves thickness at each accepted site and inside supported interpolation triangles unchanged. Beyond finite measured bearings it averages three assumptions with equal weight. A deposit can retain the original broad field, or continue preferentially along one broad or narrower fan. A site carrying only an upper limit constrains its own location and supplies no positive ray.

[Near-observation field](local.png) · [Source region](source.png) · [Whole-Earth map](world.png) · [Sector-by-sector isopach areas](bearing/sector-areas.csv) · [All crossings along eight bearings](bearing/ray-crossings.json)

## What the unobserved directions contribute

| Continuation assumption | Whole-deposit median [conditional 95%], km³ |
|---|---:|
| Broad parent continuation | 1,480.4 [508.5–40,136.3] |
| Unknown fan, 90° half-thickness angle | 1,008.2 [413.7–23,647.8] |
| Unknown fan, 47° half-thickness angle | 805.6 [374.1–16,525.6] |
| Enhanced narrow hidden fan, separate sensitivity | 1,206.1 [457.6–42,645.3] |

The three principal assumptions allow local thickness up to the parent prediction in the same draw. The enhanced-fan sensitivity allows extra ash along a hidden axis. Changing the angular spread of measured reach also changes the inferred mass.

| Angle at which measured reach halves | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| 11.25° | 914.4 [314.4–26,030.3] | Passed |
| 45° | 1,270.9 [470.0–34,457.4] | Warning |

The median fraction inside the screened measurement footprint is **2.8%**. The median fraction beyond the farthest exact site is **10.6%**. Other mass lies in the intervening gaps and the unmeasured proximal cap. These are sampling classes.

## Prediction and uncertainty

The equal-region held-out log score is **1.632**, compared with 1.632 for the original geometry. Lower scores indicate better predictions. The 95% predictive interval covered **95.2%** of exact held-out thicknesses. Every buffered training fold builds its own bearing support. Censored sites contribute their reported limit events to the score.

Across 13 held-out readings with gap exposure of at least 0.5, mean negative log score was 2.045 for the principal field and 2.088 for its parent. This subset weights individual sites equally.

The shared contour audit identified no qualifying standard-policy band. [Complete decisions and geographic deletions](complete-contour-audit.json)

| Spherical prior | Whole-deposit median [conditional 95%], km³ | Numerical checks |
|---|---:|---|
| baseline | 1,069.7 [402.9–27,836.6] | Passed |
| diffuse tail | 1,118.8 [396.0–25,212.1] | volume draw split warning |
| wide geography | 2,636.3 [543.6–147,821.0] | Passed |
| volume prior shift | 1,218.1 [419.6–49,587.6] | Passed |
| one fan | 158,293.0 [2,176.2–85,084,664.8] | volume draw split warning |
| five fans | 1,219.4 [411.4–28,624.8] | volume draw split warning |
| gaussian errors | 1,041.3 [402.6–25,865.4] | volume draw split warning |
| diffuse mixture weights | 826.5 [379.9–4,928.9] | Passed |

At 384 parent draws per chain, the following resolved contour areas still differed by at least 10% between draw halves: S 10 cm (10%), SW 10 cm (18%).

All four geographic training fits passed their sampling checks.
The tail and overall scale can remain weakly identified even when numerical checks pass. The reference-amplitude prior induces a prior on retained volume. Reworking, incomplete sampling and source attribution can add uncertainty beyond these conditional intervals.

[Prior sensitivity and mass partition](volume.png) · [Derived-volume chains](sampling.png) · [Numerical report](report.json) · [Held-out sites](heldout_predictions.csv) · [Whole-deposit draws](volume_draws.csv.gz) · [Joint sector and support-class volumes](bearing/baseline.npz) · [Original spherical fit and all attempts](bearing/parent-report.json)
