# B-Tm: empirical methods on one regional footprint

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**64 reported measurements and 3 reported minima fitted as points, 0 lower-censored fits, 4 upper limits.** The observation hull covers 796,162 km². The automatic validation buffer is 23.22 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Principal median-map volume inside hull (km³) | 24.43 | 51.55 |
| Integrated Q25–Q75 thickness (km³) | 19.44–30.74 | 32.26–91.21 |
| Outer equal-region negative log score ↓ | 1.485 | 4.184 |
| Outer log-thickness RMSE ↓ | 1.071 | 2.591 |
| Outer 95% observation-interval coverage | 89.6% | 82.1% |

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
      "weight": 1.0,
      "model": {
        "candidate": {
          "tail": "exponential",
          "length_factor": 1.5,
          "noise_ratio": 1.5
        },
        "beta": [
          3.7679942581448507,
          54.52798794319383,
          51.798889885041355,
          0.6143980321777146
        ],
        "radius_scale_km": 995.1921459120897,
        "length_km": 519.2060038733712,
        "amplitude_log": 0.7605830945626616,
        "noise_log": 1.1408746418439923,
        "jitter": 5.784866437345147e-10,
        "trend_on_constraint_boundary": false,
        "uncertainty": "conditional Laplace"
      }
    }
  ],
  "dominant_tail": "exponential",
  "observation_discrepancy_variance_log": 0.0,
  "trend_on_constraint_boundary": false,
  "uncertainty": "Predictive mixture; includes disagreement among retained candidates, conditional on each fit and estimated weights. Fitted-parameter uncertainty is not integrated."
}
```

### Trend-regularized cubic spline

```json
{
  "method": "spline",
  "candidate": {
    "tail": "exponential",
    "knot_intervals": 6,
    "tension": 0.0,
    "smoothing": 0.0001,
    "trend_strength": 0.0,
    "noise_ratio": 0.75
  },
  "bounds_km": [
    -190.76417207152855,
    -738.3045410473977,
    1362.3077833771963,
    515.5254184480947
  ],
  "knot_intervals_xy": [
    6,
    5
  ],
  "n_coefficients": 72,
  "noise_log": 0.570437320921997,
  "amplitude_log": 0.7605830945626626,
  "trend_beta": [
    3.767994258160416,
    54.52798794349392,
    51.79888988532627,
    0.6143980322222754
  ],
  "trend_radius_scale_km": 995.1921459120897,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 13.922662398089265,
  "newton_decrement": 9.037566068273991e-13,
  "newton_iterations": 5,
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
- GP ensemble joint-draw quadrature needs further refinement; inspect its grid checks before using intervals.
- GP ensemble: held-out 95% observation coverage is only 89.6%; uncertainty is undercovered.
- Trend-regularized cubic spline: the analytic expected volume exceeds the 97.5th percentile of complete deposits. Rare thick fields influence the conditional uncertainty calculation.
- Trend-regularized cubic spline: held-out 95% observation coverage is only 82.1%; uncertainty is undercovered.

## Reproduce and inspect

```sh
tephra-isopachs compare B-Tm --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
