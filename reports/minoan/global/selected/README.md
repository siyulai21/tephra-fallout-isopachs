# Minoan: deposit reconstruction

The analysis used **92 thickness measurements** and 0 documented upper limits and 9 lower bounds, excluding 0 legacy pseudo rows. Spatial validation selected the power thinning family within the directional-trend-plus-spatial-residual model.

## Primary result: complete-Earth fallout

**Expected global bulk fallout: 86.1 km³.** The integral covers the complete WGS84 surface, independently of the display extent or diagnostic rectangle. Numerical refinement and angular-phase checks passed.

The integral of the global median thickness surface is 53.6 km³. 50.1% of expected volume lies beyond the farthest sampled source-distance (1,156 km on the authalic sphere); 6.6% lies on the opposite hemisphere. These fractions describe extrapolation, not measured deposition.

The other fitted thinning family gives **40.3 km³** globally. This comparison tests tail assumptions; it is not a confidence interval.

The result is conditional on the fitted thickness model and its global extrapolation. No global probability interval is estimated. The regional simulation intervals below or in JSON do not apply globally. It is bulk fallout volume; DRE and other eruption products require separate evidence. [Global numerical record](global_volume.json) · [Global volume table](global_volume.csv)

## Historical regional diagnostic


| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | 44.2 km³ |
| Integral of the median thickness surface | 29.3 km³ |
| Expected volume outside the observation hull | 32% |
| Expected volume after doubling both region widths | 55.0 km³ |
| Last numerical refinement change | 0.15% |

Closed median isopachs were resolved at 0.3, 1, 3, 10, 30 cm. Levels 0.1 cm remain beyond the map limit and were not artificially closed.

Nested geographic holdouts covered 97% of exact measurements with nominal 95% intervals; natural-log RMSE was 1.10.

| Geographic holdout metric | Value |
|---|---:|
| Negative log score (lower is better) | 1.443 |
| Equal-region negative log score | 1.443 |
| Mean 95% interval width (natural-log cm) | 5.210 |


The regional table contains finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. The stated equal-area rectangle is easting -400 to 1200 km and northing -850 to 1100 km from the supplied source reference. Numerical convergence passed the 0.5% criterion. Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: conditional Laplace. Fitted-parameter uncertainty is not integrated. Input selection preserves the supplied audit: Combined fallout: consistent reported totals minus identified turbidites; incomplete cores are lower bounds; conflicting rows excluded; identical colocated records counted once. Geological qualifications remain in the source audit; this model run does not re-audit the original publications.

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)

![Median isopachs with observations and extrapolation](isopachs.png)

![Geographic holdouts and numerical diagnostics](diagnostics.png)
