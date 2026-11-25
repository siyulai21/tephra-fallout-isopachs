# Ilopango TBJ (total primary airfall): empirical methods on one regional footprint

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**65 reported measurements and 84 reported minima fitted as points, 0 lower-censored fits, 5 upper limits, 14 two-sided intervals.** The observation hull covers 281,690 km². The automatic validation buffer is 3.14 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Principal median-map volume inside hull (km³) | 17.26 | 27.62 |
| Integrated Q25–Q75 thickness (km³) | 10.17–30.09 | 19.84–38.90 |
| Outer equal-region negative log score ↓ | 1.799 | 1.757 |
| Outer log-thickness RMSE ↓ | 1.387 | 1.338 |
| Outer 95% observation-interval coverage | 97.3% | 98.7% |

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
      "weight": 0.5510263499623442,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.25,
          "noise_ratio": 0.75
        },
        "beta": [
          5.242563254879759,
          2.113035792874556,
          0.0038419695864931203,
          -0.1129694811440175
        ],
        "radius_scale_km": 35.82241397615788,
        "length_km": 16.28517225376695,
        "amplitude_log": 1.082087082297948,
        "noise_log": 0.811565311723461,
        "jitter": 1.1709124536760862e-09,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Laplace"
      }
    },
    {
      "weight": 0.4489736500376558,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 1.5,
          "noise_ratio": 1.5
        },
        "beta": [
          5.242563254879759,
          2.113035792874556,
          0.0038419695864931203,
          -0.1129694811440175
        ],
        "radius_scale_km": 35.82241397615788,
        "length_km": 97.71103352260171,
        "amplitude_log": 1.082087082297948,
        "noise_log": 1.623130623446922,
        "jitter": 1.1709124536760862e-09,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Laplace"
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
    "knot_intervals": 12,
    "tension": 1.0,
    "smoothing": 1.0,
    "trend_strength": 1.0,
    "noise_ratio": 1.5
  },
  "bounds_km": [
    -381.2155886679035,
    -610.3498500490973,
    628.8396303660211,
    168.98406263610104
  ],
  "knot_intervals_xy": [
    12,
    9
  ],
  "n_coefficients": 180,
  "noise_log": 1.5748650839312544,
  "amplitude_log": 1.0499100559541696,
  "trend_beta": [
    4.963359406217258,
    1.8289491508446944,
    -0.02994944360849479,
    -0.33424954894105086
  ],
  "trend_radius_scale_km": 35.82241397615788,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 5.211008236467727,
  "newton_decrement": 1.1195252881596741e-20,
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
- Spline selection reaches an explored search boundary: knot_intervals. This is a bounded-grid result, not proof of a continuous optimum.

## Reproduce and inspect

```sh
tephra-isopachs compare 'Ilopango TBJ (total primary airfall)' --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
