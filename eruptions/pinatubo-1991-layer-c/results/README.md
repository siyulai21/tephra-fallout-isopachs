# Pinatubo 1991 (Layer C) fallout reconstruction

## Volumes and geographic prediction

Each volume integrates its method’s median-thickness map. The GP and spline cover the same observation hull. The other three methods cover Earth, including fallout below 0.5 cm. Conditional 95% intervals describe variation in whole-deposit volume across posterior draws.

| Method | Median-map volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---:|---:|---:|---:|
| GP ensemble | **3.45** | 2.97–4.95 | 3.023 | 87.3% |
| Cubic spline | **14.8** | 10.4–1.69e+11 (provisional) | 1.590 | 85.3% |
| [Directional ensemble](directional/README.md) | **6.42** | 4.96–11.8 (provisional) | 1.549 † | 89.3% |
| [Spherical mixture](spherical-mixture/README.md) | **13.9** | 9.22–26.1 | 0.910 † | 97.3% |
| [Bayesian nested corridors](nested-isopachs/README.md) | **8.36** | 7.42–9.6 | 3.142 † | 62.0% |

Lower scores indicate better predictions in withheld geographic regions. Coverage counts exact withheld measurements. A dagger marks a numerical or geographic validation warning. Provisional intervals retain their fit or integration warnings.

## Full deposits and near-vent detail

The outer line is 0.5 cm. Contours and observation colours double from there. All three methods use the same near-vent window for this eruption. Each curve counts the combined area of all patches at or above a thickness.

![Pinatubo 1991 (Layer C) · Directional ensemble. Full deposit, near-vent detail and thickness versus square root of area.](directional/deposit.png)

![Pinatubo 1991 (Layer C) · Spherical mixture. Full deposit, near-vent detail and thickness versus square root of area.](spherical-mixture/deposit.png)

![Pinatubo 1991 (Layer C) · Bayesian nested corridors. Full deposit, near-vent detail and thickness versus square root of area.](nested-isopachs/deposit.png)

## Inside the observation hull

![Pinatubo 1991 (Layer C) · GP ensemble and cubic spline on a common scale and the full observation hull](isopachs.png)

The dashed boundary encloses all accepted sites. Gaps inside it can still be poorly constrained.

[Volume and geographic prediction plot](overview.png)

## Regional fit details

All methods use the same accepted observations, geographic holdouts and integration area. No method is designated preferred.

**150 exact measurements, 28 lower bounds, 30 upper limits.** The observation hull covers 279,806 km². The automatic validation buffer is 1.84 km.

![Matched median isopachs](isopachs.png)

The full observation hull and every accepted location are in view. The coloured field is clipped to the exact hull; fine coastlines are context only. [Map frame and rendering provenance](map-display.json).

| Quantity | GP ensemble | Trend-regularized cubic spline |
|---|---:|---:|
| Principal median-map volume inside hull (km³) | 3.45 | 14.82 |
| Conditional 95% volume interval (km³) | 2.97–4.95 | 10.40–168688762652.92 |
| Outer equal-region negative log score ↓ | 3.023 | 1.590 |
| Outer log-thickness RMSE ↓ | 7.106 | 7.233 |
| Outer 95% observation-interval coverage | 87.3% | 85.3% |

Lower predictive score and RMSE are better. Coverage must be read with interval width and predictive score; wider intervals alone are not improvement.

![Withheld predictions and regional volumes](diagnostics.png)

## Scope

The footprint is the convex hull of measured/bounded locations, with no source point or invented observations added. It can contain large unsampled gaps and can exclude proximal deposits. These are **regional bulk fallout volumes, not total eruption volumes or dense-rock equivalents**. No global extrapolation enters this comparison.

Volume integrates pointwise median thickness over the observation hull. The interval uses the 2.5th and 97.5th percentiles of spatially correlated volume draws.

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
          4.46528507983074,
          3.464115251912443,
          -1.3820655105201594,
          -0.483244728168785
        ],
        "radius_scale_km": 31.316630953439937,
        "length_km": 66.63195536609143,
        "amplitude_log": 0.64756476655253,
        "noise_log": 0.9713471498287951,
        "jitter": 4.193401268802327e-10,
        "trend_on_constraint_boundary": true,
        "uncertainty": "conditional Laplace"
      }
    }
  ],
  "dominant_tail": "power",
  "observation_discrepancy_variance_log": 2.436322670450779,
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
    "smoothing": 0.0001,
    "trend_strength": 0.01,
    "noise_ratio": 0.75
  },
  "bounds_km": [
    -999.4560309065577,
    -408.2282413420416,
    123.76605171399694,
    137.6115034530784
  ],
  "knot_intervals_xy": [
    6,
    3
  ],
  "n_coefficients": 54,
  "noise_log": 0.3730179325892552,
  "amplitude_log": 0.4973572434523402,
  "trend_beta": [
    3.774086491085778,
    2.671732398483353,
    -1.1451682481927397,
    -0.7687753132936445
  ],
  "trend_radius_scale_km": 31.316630953439937,
  "trend_on_constraint_boundary": false,
  "effective_df_exact": 19.729540608194018,
  "newton_decrement": 6.591041447737634e-18,
  "newton_iterations": 8,
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
- GP ensemble: held-out 95% observation coverage is only 87.3%; uncertainty is undercovered.
- Trend-regularized cubic spline: the analytic expected volume exceeds the 97.5th percentile of complete deposits. Rare thick fields influence the conditional uncertainty calculation.
- Trend-regularized cubic spline regional integration did not reach the stated refinement tolerance.
- Trend-regularized cubic spline joint-draw quadrature needs further refinement; inspect its grid checks before using intervals.
- Trend-regularized cubic spline: held-out 95% observation coverage is only 85.3%; uncertainty is undercovered.

## Reproduce and inspect

```sh
tephra-isopachs compare 'Pinatubo 1991 (Layer C)' --output results/new-regional-run
```

[Complete numerical report](report.json) · [Withheld predictions](heldout_predictions.csv) · [Accepted observation semantics](accepted_observations.csv) · [Selection tables](selection/README.md)

The JSON records input/source hashes, package versions, all fold memberships, regional integration refinements, conditional-volume sampling precision and geographic-deletion sensitivity. Historical GP global calculations remain separate.

## Which parts are actually supported?

Read the [local support and complete-contour audit](support/README.md) before extrapolating these hull volumes. It screens unsampled gaps, tests closure under geographic deletion, and separates interpolation from proximal/distal assumptions.
