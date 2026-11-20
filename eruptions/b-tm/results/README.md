# B-Tm fallout reconstruction

## Four fits with interpreted archive limits

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at B-Tm](four-method-evidence.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. On the directional map, white dashes show conditional continuation. Solid white segments connect thicknesses bracketed by measured sites in a supported triangle. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **29.59** | 18.88–46.13 | 1.459 | 89.1% |
| Cubic spline | Hull median map **50.75** · volume median 64.64 · unstable mean 6.1e+11 | 33.03–664.44 | 4.295 | 75.0% |
| [Directional ensemble · uncertain fan direction](directional/README.md) | Whole-Earth median map **33.6** · volume median 37.2 | 23.5–77.1 | 1.243 † | 93.8% |
| [Spherical mixture](spherical-mixture/README.md) | Whole-Earth volume median **49.1** | 30.7–341.7 | 1.492 † | 93.8% |

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval. † A geographic fit had a numerical warning or incomplete scoring.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **13.9%** of its volume. The directional volume interval comes from complete simulated deposits. The regional intervals condition on fitted settings.

<details>
<summary>Compare the two whole-Earth maps</summary>

| Uncertain fan direction | Spherical mixture |
|---|---|
| ![Uncertain-fan-direction median thickness over the Earth for B-Tm](directional/azimuth/world.png) | ![Spherical-mixture median thickness over the Earth for B-Tm](spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>

With the same observations and structure weights, fixing the fan directions gives a median-map volume of **34.6 km³**.

<details>
<summary>View the directional deposit extent and source area</summary>

![Principal uncertain-fan-direction median isopachs for B-Tm](directional/azimuth/isopachs.png)

</details>

## Directional assumptions

Both fits include the same interpreted archive limits. The measured-fan sensitivity fixes the directions of the continuation to the sampled bearings.

| Fan direction | Evidence assumption | Median-map volume (km³) | Complete-deposit median (km³) | Conditional 95% interval (km³) |
|---|---|---:|---:|---:|
| Measured fan directions | Thicknesses and bounds | 34.2 | 36.9 | 23.6–62.7 |

<details>
<summary>View agreement between the directional sensitivities</summary>

![Agreement on median isopach extent](directional/bearing/isopach-agreement.png)

</details>

[Measured-fan maps, held-out tests and site evidence](directional/bearing/README.md)

[Volume and geographic prediction plot](overview.png)

## Regional fit details

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**64 exact measurements, 3 lower bounds, 4 upper limits.** The observation hull covers 796,162 km². The automatic validation buffer is 23.22 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Expected regional bulk fallout (km³) | 29.59 | 608794465625.21 |
| Integral of median thickness surface (km³) | 27.64 | 50.75 |
| Conditional 95% volume interval (km³) | 18.88–46.13 | 33.03–664.44 |
| Outer equal-region negative log score ↓ | 1.459 | 4.295 |
| Outer log-thickness RMSE ↓ | 1.085 | 2.202 |
| Outer 95% observation-interval coverage | 89.1% | 75.0% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

The mean volume integrates expected thickness. The median-surface integral is a separate quantity; it is not the median volume. Volume intervals come from complete correlated surface draws, not sums of pointwise interval endpoints.

## Fitted methods

The Trend-regularized cubic spline analytical mean lies above the 97.5th percentile of the complete-surface volumes. Rare thick surfaces dominate that expectation. The complete-volume median is 64.64 km³ and its conditional interval is 33.03–664.44 km³.

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
          3.9177286079299813,
          55.40464843434249,
          52.48647992938348,
          0.8300496222510045
        ],
        "radius_scale_km": 1006.0864065815343,
        "length_km": 475.50338644035037,
        "amplitude_log": 0.7705951264423287,
        "noise_log": 1.1558926896634931,
        "jitter": 5.938168488966687e-10,
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
  "noise_log": 0.5779463448317463,
  "amplitude_log": 0.7705951264423284,
  "trend_beta": [
    3.9177286079299805,
    55.40464843434209,
    52.48647992938308,
    0.830049622251031
  ],
  "trend_radius_scale_km": 1006.0864065815343,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 13.46491254876172,
  "newton_decrement": 1.0480591788888921e-16,
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
- GP ensemble regional integration did not reach the stated refinement tolerance.
- GP ensemble joint-draw quadrature needs further refinement; inspect its grid checks before using intervals.
- GP ensemble: held-out 95% observation coverage is only 89.1%; uncertainty is undercovered.
- Trend-regularized cubic spline: the analytic expected volume exceeds the 97.5th percentile of complete deposits. Rare thick fields dominate the expectation. See the integral of the median map and the complete-volume quantiles.
- Trend-regularized cubic spline: held-out 95% observation coverage is only 75.0%; uncertainty is undercovered.

## Reproduce and inspect

```sh
tephra-isopachs compare B-Tm --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
