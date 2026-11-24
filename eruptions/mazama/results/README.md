# Mazama · regional fits inside the observation hull

[All five methods and field evidence](../README.md)

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**46 exact measurements, 129 lower bounds, 96 upper limits, 29 two-sided intervals.** The observation hull covers 1,549,105 km². The automatic validation buffer is 6.50 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Principal median-map volume inside hull (km³) | 299.29 | 1536.55 |
| Integrated Q25–Q75 thickness (km³) | 179.36–473.19 | 729.29–3526.70 |
| Outer equal-region negative log score ↓ | 1.399 | 1.231 |
| Outer log-thickness RMSE ↓ | 2.573 | 3.153 |
| Outer 95% observation-interval coverage | 71.7% | 89.1% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

Volume integrates pointwise median thickness over the observation hull. The endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over that same area. [Band integrals and refinements](thickness-bands.json)

## Fitted methods

### GP ensemble

```json
{
  "candidate": {
    "tail": "stacked"
  },
  "components": [
    {
      "weight": 0.7891711144968814,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.25,
          "noise_ratio": 1.5
        },
        "beta": [
          3.9858881628931484,
          7.894350704672223,
          3.6483448825416764,
          3.5131888312991424
        ],
        "radius_scale_km": 637.7091725210024,
        "length_km": 153.2996325107495,
        "amplitude_log": 1.171984019695657,
        "noise_log": 1.7579760295434856,
        "jitter": 1.3735465424219902e-09,
        "trend_on_constraint_boundary": false,
        "uncertainty": "conditional Laplace"
      }
    },
    {
      "weight": 0.21082888550311862,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 1.5,
          "noise_ratio": 1.5
        },
        "beta": [
          3.9858881628931484,
          7.894350704672223,
          3.6483448825416764,
          3.5131888312991424
        ],
        "radius_scale_km": 637.7091725210024,
        "length_km": 919.797795064497,
        "amplitude_log": 1.171984019695657,
        "noise_log": 1.7579760295434856,
        "jitter": 1.3735465424219902e-09,
        "trend_on_constraint_boundary": false,
        "uncertainty": "conditional Laplace"
      }
    }
  ],
  "dominant_tail": "power",
  "observation_discrepancy_variance_log": 4.5502881461001286,
  "trend_on_constraint_boundary": false,
  "uncertainty": "Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated."
}
```

### Trend-regularized cubic spline

```json
{
  "method": "spline",
  "candidate": {
    "tail": "power",
    "knot_intervals": 12,
    "tension": 1.0,
    "smoothing": 0.01,
    "trend_strength": 0.01,
    "noise_ratio": 1.5
  },
  "bounds_km": [
    -281.69991384821117,
    -687.4593539649622,
    992.029446515662,
    1442.157901592529
  ],
  "knot_intervals_xy": [
    7,
    12
  ],
  "n_coefficients": 150,
  "noise_log": 1.7579760295448632,
  "amplitude_log": 1.1719840196965754,
  "trend_beta": [
    3.9858892890624884,
    7.894347285122181,
    3.6483432204565136,
    3.513183928264128
  ],
  "trend_radius_scale_km": 637.7091725210024,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 15.177137384590086,
  "newton_decrement": 2.3162797978991853e-24,
  "newton_iterations": 7,
  "quadrature_order": 6,
  "uncertainty": "Conditional Laplace",
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
- GP ensemble: held-out 95% observation coverage is only 71.7%; uncertainty is undercovered.
- Trend-regularized cubic spline: the analytic expected volume exceeds the 97.5th percentile of complete deposits. Rare thick fields influence the conditional uncertainty calculation.
- Trend-regularized cubic spline joint-draw quadrature needs further refinement; inspect its grid checks before using intervals.
- Trend-regularized cubic spline: held-out 95% observation coverage is only 89.1%; uncertainty is undercovered.
- Spline selection reaches an explored search boundary: knot_intervals. This is a bounded-grid result, not proof of a continuous optimum.

## Reproduce and inspect

```sh
tephra-isopachs compare Mazama --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
