# Mazama: complete-Earth fallout volume

**Expected global bulk fallout: 178.2 km³**, conditional on the selected thickness model. The calculation covers the complete physical WGS84 Earth and is independent of map extent, contour levels and any user-selected calculation rectangle. The distance-based tail fraction does not detect unsampled directions or internal gaps; a small fraction is not proof of well-constrained volume.

The accepted input contains **45 thickness measurements**. Original legacy pseudo-observations remain excluded. These source collections have not received a new geological audit.

[Selected calculation](global/selected/README.md) · [Ensemble calculation](global/stacked/README.md) · [All-eruption comparison](../global-fallout/README.md) · [Method](../../docs/GLOBAL_VOLUME.md) · [Earlier regional analysis and source discussion](REGIONAL_REPORT.md)

![Mazama reconstructed median isopachs](global/selected/isopachs.png)

## What changes when the region choice is removed

| Quantity | Result |
|---|---:|
| Selected global expected bulk fallout | **178.2 km³** |
| Ensemble global expected bulk fallout | 218.1 km³ |
| Other fitted thinning family, globally | 229.7 km³ |
| Selected global median-surface integral | 163.8 km³ |
| Previous selected regional expectation | 171.7 km³ |
| Global expected volume beyond farthest observation | 4.5% |
| Global expected volume on opposite hemisphere | 0.0% |
| Four geographic-deletion/refit volumes | 162.9–315.9 km³ |

The other-family and deletion comparisons are sensitivities, **not confidence intervals**. A different fitting family can give a very different global total even when both integrals converge. The selected fit and its predictions are unchanged by the integration revision. The ensemble remains an evaluated alternative; this release does not reselect a tail to obtain a preferred volume.

The source-distance diagnostic uses arcs on the equal-area authalic sphere; the farthest accepted observation is 1,480 km from the source. The map's ochre rectangle now marks only the separate regional check. The entire Earth is integrated, including regions outside the displayed map and where no contour closes.

## Validation and remaining limits

Both global integrals pass radial/angular refinement and shifted-grid checks. The nominal numerical tolerance is 0.1%; this is an empirical integration check, not a statistical uncertainty interval. Selected-method geographic holdouts have negative log score 1.249, natural-log RMSE 0.907, and nominal 95% observation coverage 93.3%. These observation checks do not validate wholly unsampled sectors or a global volume.

![Mazama accumulation and holdout diagnostics](global/selected/diagnostics.png)

The middle panel accumulates volume across all source distances to the antipode; its dotted vertical line marks the farthest observation. The total uses the full physical Earth in both fitting-family curves.

**No global probability interval is estimated.** Conditional intervals in the regional calculation remain specific to that rectangle. Fitted-parameter, source-position, preservation and model-choice uncertainty are not fully propagated. The global extension of the planar trend/covariance is an explicit extrapolation assumption, especially far from the data. These are bulk fallout estimates; DRE and other eruption products require additional evidence.

## Reproduce

```sh
tephra-isopachs fit Mazama --output results/mazama-global --nested-cv --grid 128 --draws 1500
python -m benchmarks.global_fallout --output results/global-fallout
```

The second command reproduces both methods for all six eruptions, geographic deletions and known-truth trials. Accepted input bytes and existing historical report artifacts are preserved. [Numerical global record](global/selected/global_volume.json) · [Full run and code hashes](global/selected/report.json).
