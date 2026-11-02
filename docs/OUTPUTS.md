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

## Numerical checks and uncertainty

Midpoint quadrature is refined over 64, 128, 256 and 512 cells per side. Both integrals must change by less than 0.5% between consecutive grids; otherwise failure is reported. This empirical check is not a mathematical error bound.

Both thinning families are checked at region-width multipliers 1, 1.5 and 2. Doubling widths quadruples area. The alternative is the best validated candidate of the other family, fitted to the same data. This sensitivity range is **not a confidence interval**.

Joint log-surface draws preserve spatial correlation when approximating a conditional volume distribution. Their coarser-grid interval is secondary: parameters and model selection remain fixed, and volume-quantile convergence is not established. If its grid mean differs from the refined integral by over 1%, the HTML report withholds the interval and retains it in JSON only as a diagnostic.

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
| `virtual_predictions.npz` | Joint predicted log-thickness mean/covariance |

Large grid files are generated but excluded from the checked-in atlas. For compatibility, `report.json.volume` retains coarse joint-grid results. **Use `volume_analysis.regional` for the refined primary estimate** and `volume_analysis.domain_sensitivity` for refined comparisons.

Historical traces, source provenance and preservation have not been independently audited. Closed curves do not resolve those data limitations. Predicted values are never independent additional measurements.
