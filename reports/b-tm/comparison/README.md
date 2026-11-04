# B-Tm: two empirical methods, one regional footprint

Both methods use the same accepted observations, geographic holdouts and integration area. Neither is designated the preferred method.

**84 exact measurements, 0 lower bounds, 0 upper limits.** The observation hull covers 963,645 km². The automatic validation buffer is 24.95 km.

![Matched median isopachs](isopachs.png)

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 30.46 | 40.16 |
| Integral of median thickness surface (km³) | 27.12 | 32.93 |
| Conditional 95% volume interval (km³) | 23.22–41.28 | 22.81–86.18 |
| Outer equal-region negative log score ↓ | 1.331 | 1.814 |
| Outer log-thickness RMSE ↓ | 0.926 | 1.715 |
| Outer 95% observation-interval coverage | 97.6% | 90.5% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

The mean volume integrates expected thickness. The median-surface integral is a separate quantity; it is not the median volume. Volume intervals come from complete correlated surface draws, not sums of pointwise interval endpoints.

## Automatic spline fit

```json
{
  "tail": "power",
  "knot_intervals": 2,
  "tension": 1.0,
  "smoothing": 0.01,
  "trend_strength": 0.0,
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
tephra-isopachs compare B-Tm --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.
