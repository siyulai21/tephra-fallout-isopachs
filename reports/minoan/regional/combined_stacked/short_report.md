# Minoan · combined: deposit reconstruction

**Method: GP ensemble (`stacked`) · historical regional calculation.** [Method definitions](../../../../docs/METHODS.md)

The analysis used **92 thickness measurements** and 0 documented upper limits and 9 lower bounds, excluding 0 legacy pseudo rows. Spatial holdouts fitted an experimental mixture of exponential and power-law thinning candidates. Mixture quantiles and expected thickness are computed directly; whole-field draws preserve both spatial covariance and model disagreement.

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


These are finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -400 to 1200 km and northing -850 to 1100 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated. Fitted-parameter uncertainty is not integrated. Input selection preserves the supplied audit: Combined fallout: consistent reported totals minus identified turbidites; incomplete cores are lower bounds; conflicting rows excluded; identical colocated records counted once. Geological qualifications remain in the source audit; this model run does not re-audit the original publications.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
