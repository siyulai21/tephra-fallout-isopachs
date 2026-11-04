# Tambora: two empirical methods, one regional footprint

Both methods use the same accepted observations, geographic holdouts and integration area. Neither is designated the preferred method.

**26 exact measurements, 0 lower bounds, 0 upper limits.** The observation hull covers 2,980,287 km². The automatic validation buffer is 80.59 km.

![Matched median isopachs](isopachs.png)

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 97.91 | 107.69 |
| Integral of median thickness surface (km³) | 91.14 | 93.46 |
| Conditional 95% volume interval (km³) | 66.61–142.60 | 79.51–155.23 |
| Outer equal-region negative log score ↓ | 1.620 | 1.548 |
| Outer log-thickness RMSE ↓ | 1.216 | 1.186 |
| Outer 95% observation-interval coverage | 96.2% | 96.2% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

The mean volume integrates expected thickness. The median-surface integral is a separate quantity; it is not the median volume. Volume intervals come from complete correlated surface draws, not sums of pointwise interval endpoints.

## Automatic spline fit

```json
{
  "tail": "exponential",
  "knot_intervals": 4,
  "tension": 1.0,
  "smoothing": 0.01,
  "trend_strength": 1.0,
  "noise_ratio": 0.75
}
```

Cubic basis size, tension, derivative penalty, trend strength, trend family and observation noise are selected by geographic predictive scores. Every explored candidate is retained in the selection tables. Numerical quadrature points approximate an area-normalized penalty and never become observations.

## Reliability checks

- All probability intervals condition on fitted parameters and selected settings; they omit parameter/selection and preservation/sampling uncertainty.
- Four geographic outer folds evaluate the entire tuning procedure; this is development evidence, not independent external validation.
- Geographic-deletion volume ranges show refitting sensitivity and are not confidence intervals.

## Reproduce and inspect

```sh
tephra-isopachs compare Tambora --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.
