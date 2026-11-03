# Minoan: deposit reconstruction

**Method: GP ensemble (`stacked`) · complete-Earth calculation.** [Method definitions](../../../../docs/METHODS.md)

The analysis used **92 thickness measurements** and 0 documented upper limits and 9 lower bounds, excluding 0 legacy pseudo rows. Spatial holdouts fitted an experimental mixture of exponential and power-law thinning candidates. Mixture quantiles and expected thickness are computed directly; whole-field draws preserve both spatial covariance and model disagreement.

## Primary result: complete-Earth fallout

**Expected global bulk fallout: 81.3 km³.** The integral covers the complete WGS84 surface, independently of the display extent or diagnostic rectangle. Numerical refinement and angular-phase checks passed.

The integral of the global median thickness surface is 53.8 km³. 52.9% of expected volume lies beyond the farthest sampled source-distance (1,156 km on the authalic sphere); 7.0% lies on the opposite hemisphere. These fractions describe extrapolation, not measured deposition.

The other fitted thinning family gives **40.3 km³** globally. This comparison tests tail assumptions; it is not a confidence interval.

The result is conditional on the fitted thickness model and its global extrapolation. No global probability interval is estimated. The regional simulation intervals below or in JSON do not apply globally. It is bulk fallout volume; DRE and other eruption products require separate evidence. [Global numerical record](global_volume.json) · [Global volume table](global_volume.csv)

## Historical regional diagnostic


| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 39.4 km³ |
| Integral of the median thickness surface | 29.6 km³ |
| Expected volume outside the observation hull | 31% |
| Expected volume after doubling both region widths | 50.2 km³ |
| Last numerical refinement change | 0.05% |

Closed median isopachs were resolved at 0.3, 1, 3, 10, 30 cm. Levels 0.1 cm remain beyond the map limit and were not artificially closed.

Nested geographic holdouts covered 95% of exact measurements with nominal 95% intervals; natural-log RMSE was 1.12.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 1.480 |
| Equal-region negative log score | 1.481 |
| Mean 95% interval width (natural-log cm) | 4.943 |

| Retained component | Length factor | Noise ratio | Weight |
|---|---:|---:|---:|
| power | 0.75 | 0.75 | 68.4% |
| power | 1.5 | 1.5 | 31.6% |

Weights are predictive contributions, not probabilities that models are true. Extra observation-discrepancy adjustment is diagnostic only; see the numerical report.


The regional table contains finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -400 to 1200 km and northing -850 to 1100 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated. Fitted-parameter uncertainty is not integrated. Input selection preserves the supplied audit: Combined fallout: consistent reported totals minus identified turbidites; incomplete cores are lower bounds; conflicting rows excluded; identical colocated records counted once. Geological qualifications remain in the source audit; this model run does not re-audit the original publications.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
