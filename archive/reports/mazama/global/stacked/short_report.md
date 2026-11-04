# Mazama: deposit reconstruction

**Method: GP ensemble (`stacked`) · complete-Earth calculation.** [Method definitions](../../../../../docs/METHODS.md)

The analysis used **45 thickness measurements** and 0 documented upper limits and 0 lower bounds, excluding 7 legacy pseudo rows. Spatial holdouts fitted an experimental mixture of exponential and power-law thinning candidates. Mixture quantiles and expected thickness are computed directly; whole-field draws preserve both spatial covariance and model disagreement.

## Primary result: complete-Earth fallout

**Expected global bulk fallout: 218.1 km³.** The integral covers the complete WGS84 surface, independently of the display extent or diagnostic rectangle. Numerical refinement and angular-phase checks passed.

The integral of the global median thickness surface is 177.0 km³. 17.5% of expected volume lies beyond the farthest sampled source-distance (1,480 km on the authalic sphere); 1.1% lies on the opposite hemisphere. These fractions describe extrapolation, not measured deposition.

The other fitted thinning family gives **178.2 km³** globally. This comparison tests tail assumptions; it is not a confidence interval.

The result is conditional on the fitted thickness model and its global extrapolation. No global probability interval is estimated. The regional simulation intervals below or in JSON do not apply globally. It is bulk fallout volume; DRE and other eruption products require separate evidence. [Global numerical record](global_volume.json) · [Global volume table](global_volume.csv)

## Historical regional diagnostic


| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 182.5 km³ |
| Integral of the median thickness surface | 156.7 km³ |
| Expected volume outside the observation hull | 25% |
| Expected volume after doubling both region widths | 197.2 km³ |
| Last numerical refinement change | 0.00% |

Closed median isopachs were resolved at 0.1, 1, 2, 5, 10, 20, 50, 100, 200 cm. Extrapolated contour sections are dashed on the map.

Nested geographic holdouts covered 93% of exact measurements with nominal 95% intervals; natural-log RMSE was 0.93.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 1.237 |
| Equal-region negative log score | 1.234 |
| Mean 95% interval width (natural-log cm) | 3.311 |

| Retained component | Length factor | Noise ratio | Weight |
|---|---:|---:|---:|
| exponential | 0.25 | 0.25 | 20.9% |
| exponential | 0.75 | 0.75 | 20.9% |
| power | 0.25 | 0.75 | 20.6% |
| power | 1.5 | 0.75 | 37.6% |

Weights are predictive contributions, not probabilities that models are true. Extra observation-discrepancy adjustment is diagnostic only; see the numerical report.


The regional table contains finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -377.75 to 1168.75 km and northing -357.375 to 1679.47 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated. Fitted-parameter uncertainty is not integrated. Input provenance and preservation were not independently audited.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
