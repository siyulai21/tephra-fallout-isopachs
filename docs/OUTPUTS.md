# Reading maps, volumes and geographic validation

Open the eruption page for its current maps, accepted input and field audit. The principal catalogue lists published fits on the registered input. Each row records the input hash, integration area and geographic holdout results. A family awaiting a rerun on a changed dataset has no current estimate. [Eruption catalogue](../eruptions/README.md) · [Numerical results](../comparisons/principal-volumes.csv)


Model maps show each lower-only site as a measurement point at the reported minimum. Upper limits and intervals retain their own symbols. Source-audit figures use the original geological classifications. [Model-map observation symbols](methods/reported-minimum-fitting.md)

## Read the map and its volume

For each fit let $T_p(x)$ denote the pointwise posterior $p$-quantile of latent thickness in cm. All three calculations use the same model, mixture weights, coordinates and physical domain $D$. The reported bulk volume in km³ is

$$ V_p=10^{-5}\int_D T_p(x)\,dA,\qquad p\in\{0.25,0.50,0.75\}, $$

where area is in km². The map shows $T_{0.50}$ and the principal estimate is $V_{0.50}$. The conditional range runs from $V_{0.25}$ to $V_{0.75}$. The 50% probability applies at each location. Assigning a probability to total volume would also require the spatial dependence of the entire field. Sampling warnings, prior sensitivities, geographic deletion tests and 95% **observation** coverage each have separate columns or reports.

The GP and spline integrate their latent thickness distributions over the observation hull. Positive-weight triangle quadrature tests successive refinements for all three surfaces. The global models integrate each quantile around the whole WGS84 Earth, including ash thinner than the figure shows. Each global report includes a compact median grid. It recovers the published contours, area curve and volume directly. The directional posterior evaluates thickness directly. The earlier triangulated directional control preserves its original connectivity in a separate replay file. Polar interpolation is periodic in bearing and bilinear in log thickness. Two independent scrambled equal-area rules evaluate the posterior fields directly. Halving the mesh checks spatial resolution. [Support criteria and unsampled gaps](methods/supported-isopachs.md)

### Full-deposit panel

Evaluate the plotted median interpolant at every accepted model row, including measured values, bounds and two-sided intervals. Choose the **highest power of two in cm** whose footprint contains at least 95% of those rows, then apply a **1 cm minimum**. Membership uses the predicted thickness at the row coordinate. Disconnected patches count and holes stay outside. An observation on a contour counts inside. The caption reports the resulting count even when the floor prevents 95% coverage. The innermost contour is the highest doubling level that still encloses one or more modeled rows. Its receipt gives the enclosed row IDs. This display rule leaves the geological support decision to the independent contour audit. The frame includes every patch of the outer contour. The GP and spline apply the same upper-level criterion to their respective median grids inside the observation hull. The fill above each panel’s highest eligible level uses one color. Observation symbols preserve the measured or bounded thickness bracket.

### Near-vent panel and area curve

All methods fitted to one input use the same square centred on its vent. The initial window follows the local site geometry. Distances use distinct accepted coordinates, including bounded sites. Start with the eighth-nearest positive distance (or the farthest available). Cap it at three times the median of the nearest three positive distances, add 25% padding, and constrain the half-width to 1–100 km. Round upward at the leading decimal place. A vent-only input uses 1 km. Starting there, the renderer tries successively wider windows. It chooses the smallest tested square showing two eligible levels for every available method when possible. Each qualifying path must be at least one eighth of the panel width. A fit with only one eligible level can qualify with one. The title gives the half-width and number of modeled rows in view. The receipt lists the visible levels.

Observation colours use the measured thickness or reported limit, starting below 0.5 cm and doubling from 0.5 cm. Squares show two-sided ranges coloured by their lower edge. Dashed lines show conditional contours. Solid fragments, where provided, have local measurement brackets. Full maps use Natural Earth. Local maps use GSHHG 2.3.7 worldwide, including islands and lakes. Panels use native full-resolution shorelines at up to 0.3 km per display pixel, high resolution up to 1.5 km and intermediate resolution at broader scales. [Shoreline source and offline assets](../src/empirical_isopach/assets/shorelines/README.md) The curve plots thickness against the square root of the **combined** area of every patch at or above that level. The curve stops at the highest site-enclosing contour. The numerical area table also retains thinner and thicker modeled levels.

## Files behind each estimate

| File within a result directory | What it contains |
|---|---|
| `principal-volumes.csv` | Median map, integrated thickness quartiles, geographic score, coverage, numerical flags and input hash |
| `isopachs.png`, `map-display.json` | Paired regional maps on a common thickness scale and their hull limits |
| `report.json`, `thickness-bands.json` | Regional fits and fold membership, plus all three thickness integrals and quadrature refinements |
| `heldout_predictions.csv`, `selection/`, `support/` | Held-out observations, training-fold choices, unsupported holes and rejected contours |
| `<global family>/deposit.png`, `deposit.json`, `median-grid.npz`, `thickness-area.csv` | Full and near-vent maps, quartile integrals, convergence, contour membership by row and cumulative areas |
| `directional/multiscale/`, `spherical-mixture/` | Current directional and spherical posteriors, with full diagnostics and conditional uncertainty |
| `directional/extent/`, `spherical-mixture/extent/` | Earlier directional finite-spread control and current spherical continuation. The spherical packet includes eight fixed priors |
| `wrapped-angular-covariance/` | Joint regional and near-source posterior, maps at three scales, held-out beds, contour support, archive probes and whole-Earth quadrature refinements |
| `nested-isopachs/footprints/` | Dispersal-footprint posteriors, geographic holdouts, sector contributions, support and prior sensitivities |

Earlier corridor and whole-area fits, source-sensitivity trials and sampler diagnostics retain the estimands used in their original experiments. Use the principal table for current map-based estimates. [Method definitions](methods/README.md) · [Validation studies](../comparisons/README.md)
