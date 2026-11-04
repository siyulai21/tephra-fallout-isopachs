# Aso-4: deposit reconstruction

**Method: selected Gaussian process (`selected`) · complete-Earth calculation.** [Method definitions](../../../../../docs/METHODS.md)

The analysis used **87 thickness measurements** and 0 documented upper limits and 0 lower bounds, excluding 6 legacy pseudo rows. Spatial validation selected the power thinning family within the directional-trend-plus-spatial-residual model.

## Primary result: complete-Earth fallout

**Expected global bulk fallout: 2,609.3 km³.** The integral covers the complete WGS84 surface, independently of the display extent or diagnostic rectangle. Numerical refinement and angular-phase checks passed.

The integral of the global median thickness surface is 1,887.9 km³. 60.2% of expected volume lies beyond the farthest sampled source-distance (2,909 km on the authalic sphere); 15.5% lies on the opposite hemisphere. These fractions describe extrapolation, not measured deposition.

The other fitted thinning family gives **22,291.7 km³** globally. This comparison tests tail assumptions; it is not a confidence interval.

The result is conditional on the fitted thickness model and its global extrapolation. No global probability interval is estimated. The regional simulation intervals below or in JSON do not apply globally. It is bulk fallout volume; DRE and other eruption products require separate evidence. [Global numerical record](global_volume.json) · [Global volume table](global_volume.csv)

## Historical regional diagnostic


| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 646.5 km³ |
| Integral of the median thickness surface | 491.2 km³ |
| Expected volume outside the observation hull | 68% |
| Expected volume after doubling both region widths | 1,317.3 km³ |
| Last numerical refinement change | 0.01% |

Closed median isopachs were resolved at 1, 2, 5, 10, 20, 50 cm. Levels 0.1 cm remain beyond the map limit and were not artificially closed.

Nested geographic holdouts covered 79% of exact measurements with nominal 95% intervals; natural-log RMSE was 1.14.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 2.013 |
| Equal-region negative log score | 2.025 |
| Mean 95% interval width (natural-log cm) | 3.866 |


The regional table contains finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -730.373 to 3407.1 km and northing -909.873 to 2336.91 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: conditional Gaussian. Fitted-parameter uncertainty is not integrated. Input provenance and preservation were not independently audited.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
