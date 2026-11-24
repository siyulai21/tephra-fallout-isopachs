# Mazama · historical five-method comparison on 271 localities

This matched comparison uses the preceding audit with 31 measurements, 137 lower bounds and 103 upper bounds. [Current 300-locality reconstruction](../README.md) · [Historical input](../../../data/eruptions/Mazama_reviewed.csv) · [Matched numerical results](principal-volumes.csv)

## Volumes and geographic prediction

Each estimate integrates the median thickness on its map. The conditional range integrates the pointwise 25th and 75th percentiles of thickness over the same area. The 50% probability refers to thickness at each location. A single probability for total volume does not follow from these endpoints.

| Method | Median map (km³) | Q25–Q75 thickness maps (km³) | Held-out score ↓ | 95% predictive coverage |
|---|---:|---:|---:|---:|
| GP ensemble | **282** | 182–427 | 0.719 | 77.4% |
| Cubic spline | **1,107** | 541–2,597 | 1.738 | 77.4% |
| [Directional ensemble](directional/README.md) | **214** | 145–326 † | 0.530 † | 96.8% |
| [Spherical mixture](spherical-mixture/README.md) | **241** | 165–403 | 0.532 | 74.2% |
| [Bayesian nested footprints](nested-isopachs/README.md) | **510** | 320–811 | 0.327 † | 93.5% |

Lower scores indicate better geographic prediction. Coverage counts exact withheld measurements. A dagger marks a fit, integration or validation warning.

## Full deposits and near-vent detail

Each full map follows the thickest power-of-two contour enclosing at least 95% of the fitted locations, with a 1 cm minimum. The title gives the actual count, including when that minimum prevents 95% coverage. Global volumes include the thinner tail. The innermost line is the thickest doubling level enclosing an accepted location. The near-vent window expands from the proximal sites until eligible contours are visible. The curve sums all patches at each displayed thickness.

![Mazama · Directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](directional/deposit.png)

![Mazama · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](spherical-mixture/deposit.png)

![Mazama · Bayesian nested footprints. Full deposit, near-vent detail and thickness versus square root of area.](nested-isopachs/deposit.png)

## Inside the observation hull

![Mazama · GP ensemble and cubic spline on a common scale and the full observation hull](isopachs.png)

The GP and spline integrate over the dashed hull. Gaps inside it can still be poorly constrained.

## Regional fit details

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**31 exact measurements, 137 lower bounds, 103 upper limits.** The observation hull covers 1,548,745 km². The automatic validation buffer is 6.40 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Principal median-map volume inside hull (km³) | 281.79 | 1107.29 |
| Integrated Q25–Q75 thickness (km³) | 181.51–426.78 | 541.43–2,596.78 |
| Outer equal-region negative log score ↓ | 0.719 | 1.738 |
| Outer log-thickness RMSE ↓ | 2.661 | 3.202 |
| Outer 95% observation-interval coverage | 77.4% | 77.4% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

Volume integrates pointwise median thickness over the observation hull. The endpoints integrate the pointwise Q25 and Q75 thickness surfaces over the same hull. [Band integrals and refinements](thickness-bands.json)

## Fitted methods

### GP ensemble

```json
{
  "candidate": {
    "tail": "stacked"
  },
  "components": [
    {
      "weight": 0.8360412285554798,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.25,
          "noise_ratio": 1.5
        },
        "beta": [
          3.7698958174991946,
          5.895090057552291,
          1.3577586297570094,
          1.121962121157135
        ],
        "radius_scale_km": 654.6816725908722,
        "length_km": 178.0996124849799,
        "amplitude_log": 1.1591962625572947,
        "noise_log": 1.738794393835942,
        "jitter": 1.3437359751268007e-09,
        "trend_on_constraint_boundary": false,
        "uncertainty": "conditional Laplace"
      }
    },
    {
      "weight": 0.16395877144452026,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 1.5,
          "noise_ratio": 1.5
        },
        "beta": [
          3.7698958174991946,
          5.895090057552291,
          1.3577586297570094,
          1.121962121157135
        ],
        "radius_scale_km": 654.6816725908722,
        "length_km": 1068.5976749098795,
        "amplitude_log": 1.1591962625572947,
        "noise_log": 1.738794393835942,
        "jitter": 1.3437359751268007e-09,
        "trend_on_constraint_boundary": false,
        "uncertainty": "conditional Laplace"
      }
    }
  ],
  "dominant_tail": "power",
  "observation_discrepancy_variance_log": 2.1094843322401102,
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
    "knot_intervals": 12,
    "tension": 1.0,
    "smoothing": 0.01,
    "trend_strength": 1.0,
    "noise_ratio": 1.5
  },
  "bounds_km": [
    -281.69991384821134,
    -687.459353964966,
    992.0294465156619,
    1442.1579015925247
  ],
  "knot_intervals_xy": [
    7,
    12
  ],
  "n_coefficients": 150,
  "noise_log": 1.7393141610312586,
  "amplitude_log": 1.159542774020839,
  "trend_beta": [
    3.4002044931325512,
    3.727225055188904,
    1.0728190885519286,
    0.8755768102428194
  ],
  "trend_radius_scale_km": 654.6816725908722,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 11.429994912484126,
  "newton_decrement": 3.8658528816012886e-22,
  "newton_iterations": 6,
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
- GP ensemble: held-out 95% observation coverage is only 77.4%; uncertainty is undercovered.
- Trend-regularized cubic spline: held-out 95% observation coverage is only 77.4%; uncertainty is undercovered.
- Spline selection reaches an explored search boundary: knot_intervals. This is a bounded-grid result, not proof of a continuous optimum.

## Reproduce and inspect

```sh
tephra-isopachs compare Mazama --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
