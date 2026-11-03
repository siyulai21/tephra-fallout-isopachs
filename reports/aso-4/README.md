# Aso-4: complete-Earth fallout volume

**Method: Gaussian process · current complete-Earth results.** [All eruptions](../README.md) · [Methods explained](../../docs/METHODS.md) · [Regional history](regional/README.md)

**Expected global bulk fallout: 2,609.3 km³**, conditional on the selected thickness model. The calculation covers the complete physical WGS84 Earth and is independent of map extent, contour levels and any user-selected calculation rectangle. **More than half of the selected global estimate lies beyond the farthest observation. Treat this as a strongly extrapolation-dependent model estimate.**

The accepted input contains **87 thickness measurements**. Original legacy pseudo-observations remain excluded. These source collections have not received a new geological audit.

[Selected GP calculation](global/selected/README.md) · [GP ensemble calculation](global/stacked/README.md) · [All-eruption comparison](../global-fallout/README.md) · [Method](../../docs/methods/global-volume.md) · [Earlier regional analysis and source discussion](regional/README.md)

![Aso-4 reconstructed median isopachs](global/selected/isopachs.png)

## What changes when the region choice is removed

| Quantity | Result |
|---|---:|
| Selected global expected bulk fallout | **2,609.3 km³** |
| Ensemble global expected bulk fallout | 2,618.8 km³ |
| Other fitted thinning family, globally | 22,291.7 km³ |
| Selected global median-surface integral | 1,887.9 km³ |
| Previous selected regional expectation | 646.5 km³ |
| Global expected volume beyond farthest observation | 60.2% |
| Global expected volume on opposite hemisphere | 15.5% |
| Four geographic-deletion/refit volumes | 1,379.5–3,582.4 km³ |

The other-family and deletion comparisons are sensitivities, **not confidence intervals**. A different fitting family can give a very different global total even when both integrals converge. The selected fit and its predictions are unchanged by the integration revision. The ensemble remains an evaluated alternative; this release does not reselect a tail to obtain a preferred volume.

The source-distance diagnostic uses arcs on the equal-area authalic sphere; the farthest accepted observation is 2,909 km from the source. The map's ochre rectangle now marks only the separate regional check. The entire Earth is integrated, including regions outside the displayed map and where no contour closes.

The alternative exponential model gives about 22,292 km³ globally, and selected observation coverage is only 79.3%. **Neither numerical convergence nor the choice of the preferred candidate establishes a reliable Aso-4 global volume.** The extreme comparison is retained to expose the extrapolation failure.

## Validation and remaining limits

Both global integrals pass radial/angular refinement and shifted-grid checks. The nominal numerical tolerance is 0.1%; this is an empirical integration check, not a statistical uncertainty interval. Selected-method geographic holdouts have negative log score 2.013, natural-log RMSE 1.137, and nominal 95% observation coverage 79.3%. These observation checks do not validate wholly unsampled sectors or a global volume.

![Aso-4 accumulation and holdout diagnostics](global/selected/diagnostics.png)

The middle panel accumulates volume across all source distances to the antipode; its dotted vertical line marks the farthest observation. The total uses the full physical Earth in both fitting-family curves.

**No global probability interval is estimated.** Conditional intervals in the regional calculation remain specific to that rectangle. Fitted-parameter, source-position, preservation and model-choice uncertainty are not fully propagated. The global extension of the planar trend/covariance is an explicit extrapolation assumption, especially far from the data. These are bulk fallout estimates; DRE and other eruption products require additional evidence.

## Reproduce

```sh
tephra-isopachs fit Aso-4 --output results/aso-4-global --nested-cv --grid 128 --draws 1500
python -m benchmarks.global_fallout --output results/global-fallout
```

The second command reproduces both methods for all six eruptions, geographic deletions and known-truth trials. Accepted input bytes and existing historical report artifacts are preserved. [Numerical global record](global/selected/global_volume.json) · [Full run and code hashes](global/selected/report.json).
