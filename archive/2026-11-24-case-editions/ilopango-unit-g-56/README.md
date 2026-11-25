# Ilopango Unit G · 56-locality regional fits

[Edition history](../README.md) · [Current case](../../../eruptions/ilopango-tbj/README.md)

[All available methods on the eruption page](../../../eruptions/ilopango-tbj/README.md)

The GP and cubic spline use the same accepted observations, geographic holdouts and observation hull.

**52 exact measurements, 4 lower bounds, 0 upper limits.** The observation hull covers 166,961 km². The automatic validation buffer is 8.11 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Principal median-map volume inside hull (km³) | 23.31 | 25.67 |
| Integrated Q25–Q75 thickness (km³) | 18.39–29.70 | 21.29–31.01 |
| Outer equal-region negative log score ↓ | 1.343 | 1.462 |
| Outer log-thickness RMSE ↓ | 0.964 | 1.053 |
| Outer 95% observation-interval coverage | 98.1% | 92.3% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

Volume integrates pointwise median thickness over the observation hull. The endpoints integrate the pointwise 25th and 75th percentiles of latent thickness on the same hull. [Band integrals and refinements](thickness-bands.json)

## Fitted methods

### GP ensemble

```json
{
  "candidate": {
    "tail": "stacked"
  },
  "components": [
    {
      "weight": 1.0,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 1.5,
          "noise_ratio": 1.5
        },
        "beta": [
          5.877882565797928,
          2.3646239542705834,
          0.014532660036742534,
          -0.36433322738101886
        ],
        "radius_scale_km": 36.57850657147445,
        "length_km": 159.7258596900914,
        "amplitude_log": 0.6824933387478334,
        "noise_log": 1.02374000812175,
        "jitter": 4.657971574351648e-10,
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
    "knot_intervals": 4,
    "tension": 1.0,
    "smoothing": 1e-08,
    "trend_strength": 100.0,
    "noise_ratio": 1.5
  },
  "bounds_km": [
    -359.99334443125537,
    -349.72718466324744,
    183.1725013964103,
    156.57345952248915
  ],
  "knot_intervals_xy": [
    4,
    4
  ],
  "n_coefficients": 49,
  "noise_log": 0.9827594136709957,
  "amplitude_log": 0.6551729424473305,
  "trend_beta": [
    5.63400524303206,
    2.1459918386242784,
    0.024895812665345022,
    -0.4568280516334591
  ],
  "trend_radius_scale_km": 36.57850657147445,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 3.335974118041806,
  "newton_decrement": 1.325114647418844e-18,
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
- Spline selection reaches an explored search boundary: smoothing. This is a bounded-grid result, not proof of a continuous optimum.

## Reproduce and inspect

```sh
tephra-isopachs compare data/eruptions/Ilopango_TBJ_Unit_G.csv --regional-only --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
