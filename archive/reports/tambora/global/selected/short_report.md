# Tambora: deposit reconstruction

**Method: selected Gaussian process (`selected`) · complete-Earth calculation.** [Method definitions](../../../../../docs/METHODS.md)

The analysis used **26 thickness measurements** and 0 documented upper limits and 0 lower bounds, excluding 2 legacy pseudo rows. Spatial validation selected the power thinning family within the directional-trend-plus-spatial-residual model.

## Primary result: complete-Earth fallout

**Expected global bulk fallout: 153.5 km³.** The integral covers the complete WGS84 surface, independently of the display extent or diagnostic rectangle. Numerical refinement and angular-phase checks passed.

The integral of the global median thickness surface is 135.9 km³. 7.1% of expected volume lies beyond the farthest sampled source-distance (1,809 km on the authalic sphere); 0.2% lies on the opposite hemisphere. These fractions describe extrapolation, not measured deposition.

The other fitted thinning family gives **180.0 km³** globally. This comparison tests tail assumptions; it is not a confidence interval.

The result is conditional on the fitted thickness model and its global extrapolation. No global probability interval is estimated. The regional simulation intervals below or in JSON do not apply globally. It is bulk fallout volume; DRE and other eruption products require separate evidence. [Global numerical record](global_volume.json) · [Global volume table](global_volume.csv)

## Historical regional diagnostic


| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 136.3 km³ |
| Integral of the median thickness surface | 122.1 km³ |
| Expected volume outside the observation hull | 28% |
| Expected volume after doubling both region widths | 149.3 km³ |
| Last numerical refinement change | 0.16% |

Closed median isopachs were resolved at 0.1, 1, 2, 5, 10, 20, 50, 100 cm. Extrapolated contour sections are dashed on the map.

Nested geographic holdouts covered 92% of exact measurements with nominal 95% intervals; natural-log RMSE was 1.48.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 1.975 |
| Equal-region negative log score | 1.919 |
| Mean 95% interval width (natural-log cm) | 4.378 |


The regional table contains finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -2510.44 to 2088.57 km and northing -702.917 to 1833.21 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: conditional Gaussian. Fitted-parameter uncertainty is not integrated. Input provenance and preservation were not independently audited.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
