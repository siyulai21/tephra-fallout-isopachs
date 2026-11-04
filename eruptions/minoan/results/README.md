# Minoan: two empirical methods, one regional footprint

Both methods use the same accepted observations, geographic holdouts and integration area. Neither is designated the preferred method.

**92 exact measurements, 9 lower bounds, 0 upper limits.** The observation hull covers 760,879 km². The automatic validation buffer is 15.34 km.

![Matched median isopachs](isopachs.png)

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 27.21 | 24.78 |
| Integral of median thickness surface (km³) | 22.05 | 24.78 |
| Conditional 95% volume interval (km³) | 18.21–40.34 | 24.79–25.00 |
| Outer equal-region negative log score ↓ | 1.451 | 1.601 |
| Outer log-thickness RMSE ↓ | 1.095 | 1.282 |
| Outer 95% observation-interval coverage | 96.7% | 96.7% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

The mean volume integrates expected thickness. The median-surface integral is a separate quantity; it is not the median volume. Volume intervals come from complete correlated surface draws, not sums of pointwise interval endpoints.

## Automatic spline fit

```json
{
  "tail": "exponential",
  "knot_intervals": 8,
  "tension": 1.0,
  "smoothing": 1e-08,
  "trend_strength": 1000000.0,
  "noise_ratio": 1.5
}
```

Cubic basis size, tension, derivative penalty, trend strength, trend family and observation noise are selected by geographic predictive scores. Every explored candidate is retained in the selection tables. Numerical quadrature points approximate an area-normalized penalty and never become observations.

## Reliability checks

- All probability intervals condition on fitted parameters and selected settings; they omit parameter/selection and preservation/sampling uncertainty.
- Four geographic outer folds evaluate the entire tuning procedure; this is development evidence, not independent external validation.
- Geographic-deletion volume ranges show refitting sensitivity and are not confidence intervals.
- Spline selection reaches an explored search boundary: smoothing, trend_strength. This is a bounded-grid result, not proof of a continuous optimum.
- The selected spline strongly follows its estimated trend. Its conditional volume interval can be extremely narrow because trend-estimation uncertainty is omitted; inspect the geographic-deletion range rather than interpreting narrowness as established precision.

## Reproduce and inspect

```sh
tephra-isopachs compare Minoan --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.
