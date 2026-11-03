# Reading the maps and volumes

## Closed isopachs

Maps show level sets of **median predicted thickness**. The map expands on sides where a requested level reaches the frame. By default no projected coordinate exceeds 6,000 km in absolute value. This is a display limit, not a deposit boundary. If a level cannot be resolved within it, a thicker outer level is used and the unresolved level is listed.

Every resolved path must return to its start. Separate endpoints are never joined around the frame. Areas subtract holes and include separate islands. The map grid is compared with a grid of half its resolution. Features changing by over 0.5% receive a local 481×481 refinement around all globally resolved components. Remaining changes above 2% are flagged. Each area's numerical grid scope is recorded in the CSV.

Solid sections lie inside the observation convex hull; dashed sections lie outside. This is a geometric support indicator, not a confidence band. It cannot identify internal gaps. The map extent is independent of the volume region, shown by the ochre rectangle.

## Volume definitions

The primary estimate is conditional expected **bulk deposit volume in a stated finite rectangle**:

\[
E[V_D\mid\widehat\theta,y]=10^{-5}\int_D\exp\{\mu(s)+v(s)/2\}\,ds.
\]

Area is km² and thickness cm; conversion to km³ is 10⁻⁵. The default rectangle includes the vent and all non-pseudo observations, padded by 25%. `--bounds` overrides it.

The integral of the pointwise median surface is reported separately:

\[
V_{\mathrm{median\ surface},D}=10^{-5}\int_D\exp\{\mu(s)\}\,ds.
\]

This is **not** the median of random total volume. Neither estimate is dense-rock equivalent: no density or vesicularity correction is supplied.

These formulas describe the default single-component model. With `--method stacked`, expected thickness is the weighted sum of component lognormal expectations, and the median solves the mixture CDF. Exponentiating the mixture's mean log thickness or applying a lognormal formula to its moment covariance would be incorrect. See [the empirical method](EMPIRICAL_METHODS.md) for the full definitions.

## Numerical checks and uncertainty

Midpoint quadrature is refined over 64, 128, 256 and 512 cells per side. Both integrals must change by less than 0.5% between consecutive grids; otherwise failure is reported. This empirical check is not a mathematical error bound.

Both thinning families are checked at region-width multipliers 1, 1.5 and 2. Doubling widths quadruples area. The alternative is the best validated candidate of the other family, fitted to the same data. This sensitivity range is **not a confidence interval**.

For the ensemble, the main curve is the mixture and the comparison is the best candidate from the family opposite its largest total weight. The comparison can therefore already be represented within the ensemble; it is a sensitivity reference, not independent evidence or a bound.

Joint log-surface draws preserve spatial correlation when approximating a conditional volume distribution. Their coarser-grid interval is secondary: parameters and model selection remain fixed, and volume-quantile convergence is not established. If its grid mean differs from the refined integral by over 1%, the HTML report withholds the interval and retains it in JSON only as a diagnostic.

Ensemble draws also preserve retained-model disagreement by choosing one component for each entire field. Fitted component parameters and learned weights remain fixed. The extra observation-discrepancy experiment is excluded from every latent map and volume draw.

Numerical convergence does not establish geological accuracy. We do not integrate to infinite distance and label the result total erupted volume. Power tails may contribute substantial distant volume even when a positive-thickness contour closes. `volume_analysis.total_erupted_volume_km3` is therefore null.

## Output files

| File | Contents |
|---|---|
| `index.html` | Offline responsive report |
| `short_report.md` | Concise, shareable findings and limitations |
| `isopachs.png`, `.svg`, `.pdf` | Raster, scalable and print-ready maps |
| `diagnostics.png`, `.svg` | Areas, volume sensitivity and holdouts |
| `isopachs.geojson` | WGS84 contour lines; antimeridian crossings split |
| `isopach_areas.csv` | Equal-area enclosed areas and resolution checks |
| `volume_summary.csv` | Refined estimates and explicit region bounds |
| `observations_used.csv` | Accepted observations and source-row identifiers |
| `report.json` | Settings, fits, validation, warnings and source hashes |
| `predictions.csv` | Thickness summaries and support diagnostics on a grid |
| `virtual_predictions.npz` | Joint predicted log-thickness moments; ensemble exports also contain each component's weight, mean and covariance |

Large grid files are generated but excluded from the checked-in atlas. For compatibility, `report.json.volume` retains coarse joint-grid results. **Use `volume_analysis.regional` for the refined primary estimate** and `volume_analysis.domain_sensitivity` for refined comparisons.

Historical traces, source provenance and preservation have not been independently audited. Closed curves do not resolve those data limitations. Predicted values are never independent additional measurements.

## Ensemble and observation diagnostics

The ensemble report lists learned weights and their sensitivity to reweighting whole geographic folds. It compares ordinary and discrepancy-adjusted observation coverage, interval width and negative log score using outer holdouts. The adjusted version is diagnostic, with no coverage guarantee; it is not the primary interval.

In `predictions.csv`, `conditional_*` columns describe the latent thickness surface; `new_observation_*` adds the fitted residual observation noise. `diagnostic_adjusted_observation_*` also adds the training-derived discrepancy variance. These grid predictions do not assume a user-supplied measurement error for future observations. Known `LogSD` values are included when evaluating actual held-out measurements. `between_model_variance_log` reports the variance attributable to component mean disagreement. `log_sd` is a moment summary, not sufficient to reconstruct mixture quantiles.

For compatibility `mean_log` in validation remains the predicted mean log thickness; `median_log` is the exact mixture median used in plots and RMSE. The `mean_log_score` field is a negative log score (lower is better). Coverage excludes censored rows, which still contribute likelihood scores. Weight/calibration training diagnostics are distinct from nested evaluation records.
