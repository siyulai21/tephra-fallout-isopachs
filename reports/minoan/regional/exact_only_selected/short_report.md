# Minoan · exact only: deposit reconstruction

**Method: selected Gaussian process (`selected`) · historical regional calculation.** [Method definitions](../../../../docs/METHODS.md)

The analysis used **92 thickness measurements** and 0 documented upper limits and 0 lower bounds, excluding 0 legacy pseudo rows. Spatial validation selected the power thinning family within the directional-trend-plus-spatial-residual model.

| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 42.8 km³ |
| Integral of the median thickness surface | 28.1 km³ |
| Expected volume outside the observation hull | 33% |
| Expected volume after doubling both region widths | 53.6 km³ |
| Last numerical refinement change | 0.06% |

Closed median isopachs were resolved at 0.3, 1, 3, 10, 30 cm. Levels 0.1 cm remain beyond the map limit and were not artificially closed.

Nested geographic holdouts covered 97% of exact measurements with nominal 95% intervals; natural-log RMSE was 1.14.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 1.575 |
| Equal-region negative log score | 1.575 |
| Mean 95% interval width (natural-log cm) | 5.462 |


These are finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -400 to 1200 km and northing -850 to 1100 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: conditional Gaussian. Fitted-parameter uncertainty is not integrated. Input selection preserves the supplied audit: Combined main selection excluding incomplete-core lower bounds. Geological qualifications remain in the source audit; this model run does not re-audit the original publications.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
