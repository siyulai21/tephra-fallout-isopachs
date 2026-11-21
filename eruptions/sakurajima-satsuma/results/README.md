# Sakurajima Satsuma (P14) fallout reconstruction

## Four matched thickness-only fits

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at Sakurajima Satsuma (P14)](four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, solid white segments connect thicknesses bracketed by measured sites in a supported triangle. White dashes on the global maps show conditional isopachs. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **3.33** | 2.64–4.02 | 1.264 | 94.0% |
| Cubic spline | Hull mean **13.04** | 4.18–41.49 | 1.371 | 90.0% |
| [Directional ensemble · uncertain fan direction](directional/README.md) | Whole-Earth median map **7.1** · volume median 10.0 | 3.8–70.6 | 1.164 † | 96.0% |
| [Spherical mixture](spherical-mixture/README.md) | Whole-Earth volume median **11.3** | 5.1–666.5 | 1.376 | 88.0% |

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval. † A geographic fit had a numerical warning or incomplete scoring.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **2.6%** of its volume. The directional volume interval comes from complete simulated deposits. The spherical mixture reports the median integrated volume across broad and uncertain fan continuations. The regional intervals condition on fitted settings.

<details>
<summary>Compare the two whole-Earth maps</summary>

| Uncertain fan direction | Spherical mixture |
|---|---|
| ![Uncertain-fan-direction median thickness over the Earth for Sakurajima Satsuma (P14)](directional/azimuth/world.png) | ![Spherical-mixture median thickness over the Earth for Sakurajima Satsuma (P14)](spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>

With the same observations and structure weights, fixing the fan directions gives a median-map volume of **9.1 km³**.

<details>
<summary>View the directional deposit extent and source area</summary>

![Principal uncertain-fan-direction median isopachs for Sakurajima Satsuma (P14)](directional/azimuth/isopachs.png)

</details>

## Directional assumptions

The principal fit above uses quantitative thicknesses and bounds. The measured-fan model extends fallout along sampled bearings.

| Fan direction | Evidence assumption | Median-map volume (km³) | Complete-deposit median (km³) | Conditional 95% interval (km³) |
|---|---|---:|---:|---:|
| Measured fan directions | Thicknesses and bounds | 8.2 | 9.9 | 4.3–73.4 |

<details>
<summary>View agreement between the directional sensitivities</summary>

![Agreement on median isopach extent](directional/bearing/isopach-agreement.png)

</details>

[Measured-fan maps, held-out tests and site evidence](directional/bearing/README.md)

[Volume and geographic prediction plot](overview.png)

## Regional fit details

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**50 exact measurements, 5 lower bounds, 2 upper limits.** The observation hull covers 24,004 km². The automatic validation buffer is 4.23 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 3.33 | 13.04 |
| Integral of median thickness surface (km³) | 3.08 | 9.12 |
| Conditional 95% volume interval (km³) | 2.64–4.02 | 4.18–41.49 |
| Outer equal-region negative log score ↓ | 1.264 | 1.371 |
| Outer log-thickness RMSE ↓ | 0.751 | 0.862 |
| Outer 95% observation-interval coverage | 94.0% | 90.0% |

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
      "weight": 0.6147998235985108,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 0.25,
          "noise_ratio": 0.75
        },
        "beta": [
          4.913903889579061,
          2.4929627122310642,
          -0.04887611276342879,
          -0.49053274643879546
        ],
        "radius_scale_km": 27.50932370357925,
        "length_km": 9.620184733127548,
        "amplitude_log": 0.4955227376669493,
        "noise_log": 0.371642053250212,
        "jitter": 2.4554278354494826e-10,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Laplace"
      }
    },
    {
      "weight": 0.3852001764014893,
      "model": {
        "candidate": {
          "tail": "power",
          "length_factor": 1.5,
          "noise_ratio": 1.5
        },
        "beta": [
          4.913903889579061,
          2.4929627122310642,
          -0.04887611276342879,
          -0.49053274643879546
        ],
        "radius_scale_km": 27.50932370357925,
        "length_km": 57.72110839876529,
        "amplitude_log": 0.4955227376669493,
        "noise_log": 0.743284106500424,
        "jitter": 2.4554278354494826e-10,
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
    "knot_intervals": 2,
    "tension": 1.0,
    "smoothing": 1e-08,
    "trend_strength": 1.0,
    "noise_ratio": 0.75
  },
  "bounds_km": [
    -173.9670011064267,
    -124.55440141491619,
    78.93715529573417,
    99.72450519289441
  ],
  "knot_intervals_xy": [
    2,
    2
  ],
  "n_coefficients": 25,
  "noise_log": 0.35976441617200416,
  "amplitude_log": 0.4796858882293389,
  "trend_beta": [
    4.666294956564778,
    2.2056825259284674,
    -0.026380607266458116,
    -0.5730113776915264
  ],
  "trend_radius_scale_km": 27.50932370357925,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 10.506584201121612,
  "newton_decrement": 6.633535376591675e-13,
  "newton_iterations": 3,
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
tephra-isopachs compare 'Sakurajima Satsuma (P14)' --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
