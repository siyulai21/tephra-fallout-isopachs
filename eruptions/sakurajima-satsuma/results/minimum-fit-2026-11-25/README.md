# Sakurajima Satsuma (P14): empirical methods on one regional footprint

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**50 reported measurements and 5 reported minima fitted as points, 0 lower-censored fits, 2 upper limits.** The observation hull covers 24,004 km². The automatic validation buffer is 4.23 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Principal median-map volume inside hull (km³) | 3.03 | 2.97 |
| Integrated Q25–Q75 thickness (km³) | 2.43–3.81 | 2.50–3.52 |
| Outer equal-region negative log score ↓ | 1.097 | 1.103 |
| Outer log-thickness RMSE ↓ | 0.721 | 0.786 |
| Outer 95% observation-interval coverage | 96.4% | 94.5% |

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
      "weight": 0.030061666967818673,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.25,
          "noise_ratio": 0.75
        },
        "beta": [
          4.949363272063151,
          2.581884189615299,
          -0.03231708827370761,
          -0.5809850705213438
        ],
        "radius_scale_km": 27.589791784439445,
        "length_km": 10.272503739116411,
        "amplitude_log": 0.5540210127461239,
        "noise_log": 0.41551575955959297,
        "jitter": 3.069392825642408e-10,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Laplace"
      }
    },
    {
      "weight": 0.9699383330321814,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.75,
          "noise_ratio": 0.75
        },
        "beta": [
          4.949363272063151,
          2.581884189615299,
          -0.03231708827370761,
          -0.5809850705213438
        ],
        "radius_scale_km": 27.589791784439445,
        "length_km": 30.817511217349235,
        "amplitude_log": 0.5540210127461239,
        "noise_log": 0.41551575955959297,
        "jitter": 3.069392825642408e-10,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Laplace"
      }
    }
  ],
  "dominant_tail": "power",
  "observation_discrepancy_variance_log": 0.0583121333696025,
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
    "knot_intervals": 6,
    "tension": 1.0,
    "smoothing": 0.01,
    "trend_strength": 100.0,
    "noise_ratio": 1.5
  },
  "bounds_km": [
    -173.9670011064267,
    -124.55440141491619,
    78.93715529573417,
    99.72450519289441
  ],
  "knot_intervals_xy": [
    6,
    5
  ],
  "n_coefficients": 72,
  "noise_log": 0.7608124222034893,
  "amplitude_log": 0.5072082814689929,
  "trend_beta": [
    4.51075750691115,
    2.0744424770040024,
    -0.003097205408818658,
    -0.6681035264235359
  ],
  "trend_radius_scale_km": 27.589791784439445,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 4.5496108392538215,
  "newton_decrement": 2.3435606532798747e-21,
  "newton_iterations": 2,
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

## Reproduce and inspect

```sh
tephra-isopachs compare 'Sakurajima Satsuma (P14)' --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.

[Wrapped angular covariance on this input](wrapped-angular-covariance/README.md)
