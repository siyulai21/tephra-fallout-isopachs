# Tambora: deposit reconstruction

**Method: GP ensemble (`stacked`) · current complete-Earth calculation.** [Method guide](../../../../docs/METHODS.md) · [Report catalogue](../../../README.md)

The analysis used **26 thickness measurements** and 0 documented upper limits and 0 lower bounds, excluding 2 legacy pseudo rows. Spatial holdouts fitted an experimental mixture of exponential and power-law thinning candidates. Mixture quantiles and expected thickness are computed directly; whole-field draws preserve both spatial covariance and model disagreement.

## Primary result: complete-Earth fallout

**Expected global bulk fallout: 155.3 km³.** The integral covers the complete WGS84 surface, independently of the display extent or diagnostic rectangle. Numerical refinement and angular-phase checks passed.

The integral of the global median thickness surface is 136.9 km³. 6.6% of expected volume lies beyond the farthest sampled source-distance (1,809 km on the authalic sphere); 0.2% lies on the opposite hemisphere. These fractions describe extrapolation, not measured deposition.

The other fitted thinning family gives **180.0 km³** globally. This comparison tests tail assumptions; it is not a confidence interval.

The result is conditional on the fitted thickness model and its global extrapolation. No global probability interval is estimated. The regional simulation intervals below or in JSON do not apply globally. It is bulk fallout volume; DRE and other eruption products require separate evidence. [Global numerical record](global_volume.json) · [Global volume table](global_volume.csv)

## Historical regional diagnostic


| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 138.3 km³ |
| Integral of the median thickness surface | 123.3 km³ |
| Expected volume outside the observation hull | 28% |
| Expected volume after doubling both region widths | 151.4 km³ |
| Last numerical refinement change | 0.16% |

Closed median isopachs were resolved at 0.1, 1, 2, 5, 10, 20, 50, 100 cm. Extrapolated contour sections are dashed on the map.

Nested geographic holdouts covered 92% of exact measurements with nominal 95% intervals; natural-log RMSE was 1.48.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 1.970 |
| Equal-region negative log score | 1.913 |
| Mean 95% interval width (natural-log cm) | 4.335 |

| Retained component | Length factor | Noise ratio | Weight |
|---|---:|---:|---:|
| exponential | 1.5 | 1.5 | 6.7% |
| power | 0.75 | 1.5 | 93.3% |

Weights are predictive contributions, not probabilities that models are true. Extra observation-discrepancy adjustment is diagnostic only; see the numerical report.


The regional table contains finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -2510.44 to 2088.57 km and northing -702.917 to 1833.21 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated. Fitted-parameter uncertainty is not integrated. Input provenance and preservation were not independently audited.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)

![Median isopachs with observations and extrapolation](isopachs.png)

![Geographic holdouts and numerical diagnostics](diagnostics.png)
