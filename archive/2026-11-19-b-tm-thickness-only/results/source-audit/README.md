# B-Tm sensitivity to source interpretation

This source audit uses the earlier fixed-direction ensemble. Its 68-record reference integrates to 54.0 km³. The current uncertain-direction result on those observations is 44.1 km³. [Principal directional report](../directional/README.md)

Each source variant uses the same six fixed-direction structures and weights. The central estimate integrates the pointwise median thickness over Earth. The interval uses entire simulated deposits with their spatial dependence intact. It covers bulk airfall from both phases in km³.

| Reading | Sites and bounds | Median-map integral | Volume median | Conditional 95% interval | Full-data fits passing | Map integration |
|---|---:|---:|---:|---:|---:|---|
| [68-record fixed-direction reference](../directional/ensemble/report.json) | 68 | 54.0 | 73.5 | 31.8–1,211.9 | 6/6 | Pass |
| [Twelve low endpoints](range_low/result.json) | 80 | 50.8 | 67.2 | 33.2–991.1 | 6/6 | Pass |
| [Twelve high endpoints](range_high/result.json) | 80 | 59.2 | 79.1 | 38.8–1,016.5 | 6/6 | Pass |
| [Alternative Tianwenfeng and Dongfanghong stacks](correlations/result.json) | 68 | 46.8 | 64.4 | 29.7–1,222.1 | 6/6 | Pass |
| [Omit 26 compilation-carried thicknesses](omit_compilation_values/result.json) | 42 | 48.9 | 75.3 | 27.8–1,246.9 | 6/6 | Pass |

The endpoint cases restore the same twelve positive site ranges. Moving every site to the same end of its own range tests a coordinated change in reporting. The proximal case assigns 150 cm to the summit and 26 cm to Dongfanghong.

## How much depends on thicknesses carried from compilations?

Twenty-five accepted scalars have explicit compilation-only field-log qualifications. A separate Utasai scalar has primary coring coordinates and a checked sample interval, while its 4 cm thickness comes from the compilation. The excluded group includes marine stations, Utasai Bog and Lake Ichi-no-Megata. Excluding these 26 together leaves 38 measured thicknesses and the same four bounds. This selection gives a median-map integral of 48.9 km³ and a conditional whole-deposit interval of 27.8–1,246.9 km³. The other accepted sites, including approximate mapped sites, retain their original treatment. The result measures dependence on that source class and its geographic coverage. It gives no error probability for an individual record. [Exact exclusions and input hash](../../../../data/audits/B-Tm/source-class-provenance.json) · [42-row input](../../../../data/audits/B-Tm/omit_compilation_values.csv) · [Whole-deposit draws](omit_compilation_values/volume-draws.npz)

## Why the directional interval is wide

The fixed-direction reference gives every structure a weight learned from the other eruptions. Three structures allow a separate fallout fan in a poorly observed direction. Those structures assign more probability to large totals.

| Fixed-direction structure | Weight | Whole-deposit median (km³) | Conditional 95% interval (km³) |
|---|---:|---:|---:|
| narrow fans | 15.7% | 50.5 | 26.8–126.7 |
| reference | 16.6% | 55.5 | 33.7–134.7 |
| broad fans | 11.9% | 60.1 | 44.8–120.0 |
| narrow missing fan | 14.9% | 83.0 | 32.1–726.1 |
| possible missing fan | 22.9% | 100.8 | 40.4–1,548.8 |
| broad missing fan | 18.0% | 142.2 | 54.4–3,836.4 |

In the saved fixed-direction volume draws, the ensemble assigns 5.8% above 500 km³. Structures permitting a missing fan account for 100.0% of that weighted tail. The 500 km³ threshold describes where large simulated volumes come from and has no role in fitting or accepting data. [Fixed-direction whole-deposit draws](../directional/ensemble/coherent-mass.npz)

The principal fit includes geographic and source-sector holdouts. The four source scenarios are full-data refits with the target eruption excluded from ensemble calibration. Each numerical record lists the attempts, diagnostics and map refinements. The intervals condition on the corresponding data interpretation and fan model. Combining their endpoints would give a sensitivity envelope with no common coverage probability.

[Scenario table](source-scenarios.csv) · [Principal case report](../../README.md) · [Source ledger and physical interpretation](../../../../data/audits/B-Tm/README.md) · [Outermost measured sites by direction](octant-extremes.csv)

## Reproduce

From the repository root, choose fresh destinations and run

```sh
tephra-isopachs compare B-Tm --output results/new-btm-audit
python -m benchmarks.btm_source_sensitivities --work results/new-btm-source-scenarios
python -m benchmarks.verify_btm_audit
```

Each scenario in the table above has a result record and a `volume-draws.npz` alongside it. The principal fit is stored with the directional family. [Release hashes](release.json)

## Regional numerical check

The Kinashibetsu correction exposed a loss of precision in a strongly penalized spline refit. On the affected geographic fold, subtracting nearly equal penalty terms made the computed objective increase even at a stationary Newton proposal. The fitter now accepts such a proposal only if its independently recalculated Hessian and gradient meet the existing strict convergence threshold. The input values, likelihood, model families and validation folds retain their usual definitions. The check was rerun on all six eruptions. The other five exactly reproduced their published regional volumes and geographic scores. Each family completed 38 of 40 synthetic trials. The same two heavily censored training sets remained unestimable for both methods. [Matched regional results](matched-regional.csv) · [Paired comparison with previous results](spline-replay.csv) · [Synthetic trials, including failures](../../../../comparisons/regional/synthetic/comparison.csv) · [Benchmark settings](regional-benchmark-design.json)

## How does the radial thinning law affect the fit?

Six thinning laws use the same 68 observations and fixed reference fan geometry. The smooth, one-break and two-break median maps integrate to 48.4–48.6 km³. The experiment also reports predictive scores on withheld geographic regions. [Thinning-law comparison](thinning/README.md)

[Western isopachs and the independent lake and peat evidence](../directional/bearing/field-evidence.md) · [Shard-count tests for excessive narrowing](../directional/bearing/shard-transfer-comparison.csv)
