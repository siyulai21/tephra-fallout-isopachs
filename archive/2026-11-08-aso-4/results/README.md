# Aso-4: empirical methods on one regional footprint

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**87 exact measurements, 0 lower bounds, 0 upper limits.** The observation hull covers 3,632,243 km². The automatic validation buffer is 37.63 km.

![Matched median isopachs](isopachs.png)

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 207.24 | 254.11 |
| Integral of median thickness surface (km³) | 174.21 | 203.72 |
| Conditional 95% volume interval (km³) | 160.33–269.15 | 171.91–463.06 |
| Outer equal-region negative log score ↓ | 1.659 | 2.066 |
| Outer log-thickness RMSE ↓ | 1.079 | 1.261 |
| Outer 95% observation-interval coverage | 83.9% | 79.3% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

The mean volume integrates expected thickness. The median-surface integral is a separate quantity; it is not the median volume. Volume intervals come from complete correlated surface draws, not sums of pointwise interval endpoints.

## Fitted methods

### GP ensemble

```json
{
  "candidate": {
    "tail": "stacked"
  },
  "components": [
    {
      "weight": 0.8321168321978664,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.25,
          "noise_ratio": 0.75
        },
        "beta": [
          3.1230045367323087,
          2.065570389900075,
          0.06556817444968209,
          0.00039923995342634093
        ],
        "radius_scale_km": 897.1034093330701,
        "length_km": 191.0702850703864,
        "amplitude_log": 0.8249364637301771,
        "noise_log": 0.6187023477976329,
        "jitter": 6.805201691916499e-10,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Gaussian"
      }
    },
    {
      "weight": 0.16788316780213366,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.75,
          "noise_ratio": 0.75
        },
        "beta": [
          3.1230045367323087,
          2.065570389900075,
          0.06556817444968209,
          0.00039923995342634093
        ],
        "radius_scale_km": 897.1034093330701,
        "length_km": 573.2108552111592,
        "amplitude_log": 0.8249364637301771,
        "noise_log": 0.6187023477976329,
        "jitter": 6.805201691916499e-10,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Gaussian"
      }
    }
  ],
  "dominant_tail": "power",
  "observation_discrepancy_variance_log": 0.0,
  "trend_on_constraint_boundary": true,
  "uncertainty": "Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated."
}
```

### Trend-regularized cubic spline

```json
{
  "method": "spline",
  "candidate": {
    "tail": "power",
    "knot_intervals": 4,
    "tension": 1.0,
    "smoothing": 0.01,
    "trend_strength": 1.0,
    "noise_ratio": 0.75
  },
  "bounds_km": [
    -178.70979458182308,
    -476.9677020899797,
    2855.4377928649897,
    1904.0099657279166
  ],
  "knot_intervals_xy": [
    4,
    3
  ],
  "n_coefficients": 42,
  "noise_log": 0.6102200455524363,
  "amplitude_log": 0.8136267274032484,
  "trend_beta": [
    2.9015868963383684,
    0.8943578454142023,
    -0.5434271646828661,
    -0.7103246325623289
  ],
  "trend_radius_scale_km": 897.1034093330701,
  "trend_on_constraint_boundary": true,
  "effective_df_exact": 19.571395816024236,
  "newton_decrement": 1.1037459278046333e-27,
  "newton_iterations": 0,
  "quadrature_order": 6,
  "uncertainty": "Conditional Gaussian",
  "uncertainty_omits": [
    "fitted trend",
    "noise and smoothing estimation",
    "knot/trend selection",
    "sampling and preservation bias",
    "vent location error"
  ],
  "pseudo_observations": 0
}
```

Every explored candidate is retained in the selection tables. All response-dependent choices are retuned within training folds; no hand-placed pseudo-data enter the likelihood.

## Reliability checks

- All probability intervals condition on fitted parameters and selected settings; they omit parameter/selection and preservation/sampling uncertainty.
- Four geographic outer folds evaluate the entire tuning procedure; this is development evidence, not independent external validation.
- Geographic-deletion volume ranges show refitting sensitivity and are not confidence intervals.
- GP ensemble: held-out 95% observation coverage is only 83.9%; uncertainty is undercovered.
- Trend-regularized cubic spline: held-out 95% observation coverage is only 79.3%; uncertainty is undercovered.

## Reproduce and inspect

```sh
tephra-isopachs compare data/eruptions/Aso-4_tephra.csv --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.

## Whole-deposit fallout

Read the [conditional global estimate and assumption sensitivity](global/README.md). Its mass is normalized on the whole Earth; these regional partial volumes and the local contour audit retain their separate meanings.
