# Tambora: complete-Earth fallout volume

**Method: Gaussian process · current complete-Earth results.** [All eruptions](../README.md) · [Methods explained](../../docs/METHODS.md) · [Regional history](regional/README.md)

**Expected global bulk fallout: 153.5 km³**, conditional on the selected thickness model. The calculation covers the complete physical WGS84 Earth and is independent of map extent, contour levels and any user-selected calculation rectangle. The distance-based tail fraction does not detect unsampled directions or internal gaps; a small fraction is not proof of well-constrained volume.

The accepted input contains **26 thickness measurements**. Original legacy pseudo-observations remain excluded. These source collections have not received a new geological audit.

[Selected GP calculation](global/selected/README.md) · [GP ensemble calculation](global/stacked/README.md) · [All-eruption comparison](../global-fallout/README.md) · [Method](../../docs/methods/global-volume.md) · [Earlier regional analysis and source discussion](regional/README.md)

![Tambora reconstructed median isopachs](global/selected/isopachs.png)

## What changes when the region choice is removed

| Quantity | Result |
|---|---:|
| Selected global expected bulk fallout | **153.5 km³** |
| Ensemble global expected bulk fallout | 155.3 km³ |
| Other fitted thinning family, globally | 180.0 km³ |
| Selected global median-surface integral | 135.9 km³ |
| Previous selected regional expectation | 136.3 km³ |
| Global expected volume beyond farthest observation | 7.1% |
| Global expected volume on opposite hemisphere | 0.2% |
| Four geographic-deletion/refit volumes | 138.4–257.4 km³ |

The other-family and deletion comparisons are sensitivities, **not confidence intervals**. A different fitting family can give a very different global total even when both integrals converge. The selected fit and its predictions are unchanged by the integration revision. The ensemble remains an evaluated alternative; this release does not reselect a tail to obtain a preferred volume.

The source-distance diagnostic uses arcs on the equal-area authalic sphere; the farthest accepted observation is 1,809 km from the source. The map's ochre rectangle now marks only the separate regional check. The entire Earth is integrated, including regions outside the displayed map and where no contour closes.

## Validation and remaining limits

Both global integrals pass radial/angular refinement and shifted-grid checks. The nominal numerical tolerance is 0.1%; this is an empirical integration check, not a statistical uncertainty interval. Selected-method geographic holdouts have negative log score 1.975, natural-log RMSE 1.484, and nominal 95% observation coverage 92.3%. These observation checks do not validate wholly unsampled sectors or a global volume.

![Tambora accumulation and holdout diagnostics](global/selected/diagnostics.png)

The middle panel accumulates volume across all source distances to the antipode; its dotted vertical line marks the farthest observation. The total uses the full physical Earth in both fitting-family curves.

**No global probability interval is estimated.** Conditional intervals in the regional calculation remain specific to that rectangle. Fitted-parameter, source-position, preservation and model-choice uncertainty are not fully propagated. The global extension of the planar trend/covariance is an explicit extrapolation assumption, especially far from the data. These are bulk fallout estimates; DRE and other eruption products require additional evidence.

## Reproduce

```sh
tephra-isopachs fit Tambora --output results/tambora-global --nested-cv --grid 128 --draws 1500
python -m benchmarks.global_fallout --output results/global-fallout
```

The second command reproduces both methods for all six eruptions, geographic deletions and known-truth trials. Accepted input bytes and existing historical report artifacts are preserved. [Numerical global record](global/selected/global_volume.json) · [Full run and code hashes](global/selected/report.json).
