# YTT · with Maninjau: deposit reconstruction

**Method: GP ensemble (`stacked`) · historical regional calculation.** [Method definitions](../../../../../docs/METHODS.md)

The analysis used **45 thickness measurements** and 0 documented upper limits, excluding 0 legacy pseudo rows. Spatial holdouts fitted an experimental mixture of exponential and power-law thinning candidates. Mixture quantiles and expected thickness are computed directly; whole-field draws preserve both spatial covariance and model disagreement.

| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 1,477.9 km³ |
| Integral of the median thickness surface | 1,313.1 km³ |
| Expected volume outside the observation hull | 19% |
| Expected volume after doubling both region widths | 1,617.5 km³ |
| Last numerical refinement change | 0.01% |

Closed median isopachs were resolved at 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 cm. Extrapolated contour sections are dashed on the map.

Nested geographic holdouts covered 91% of observations with nominal 95% intervals; natural-log RMSE was 1.27.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 2.037 |
| Equal-region negative log score | 2.053 |
| Mean 95% interval width (natural-log cm) | 4.677 |

| Retained component | Length factor | Noise ratio | Weight |
|---|---:|---:|---:|
| exponential | 0.75 | 1.5 | 100.0% |

Weights are predictive contributions, not probabilities that models are true. Extra observation-discrepancy adjustment is diagnostic only; see the numerical report.


These are finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -6900 to 2400 km and northing -1400 to 2900 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated. Fitted-parameter uncertainty is not integrated. Input selection preserves the supplied audit: main43 plus N001/N002 with reported 23 cm each and approximate audited coordinates; explicit sensitivity override only. Geological qualifications remain in the source audit; this model run does not re-audit the original publications.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
