# Reading maps, volumes and geographic validation

Open the eruption page for the maps side by side, accepted input and field audit. A case with a revised input lists the methods fitted to that revision. Historical comparisons identify their own observation counts. [Eruption catalogue](../eruptions/README.md) · [Numerical results](../comparisons/principal-volumes.csv)

## Read the map and its volume

For each fit let $T_p(x)$ denote the pointwise posterior $p$-quantile of latent thickness in cm. All three calculations use the same model, mixture weights, coordinates and physical domain $D$. The reported bulk volume in km³ is

$$ V_p=10^{-5}\int_D T_p(x)\,dA,\qquad p\in\{0.25,0.50,0.75\}, $$

where area is in km². The map shows $T_{0.50}$ and the principal estimate is $V_{0.50}$. The conditional range runs from $V_{0.25}$ to $V_{0.75}$. The 50% probability applies at each location. Assigning a probability to total volume would also require the spatial dependence of the entire field. Sampling warnings, prior sensitivities, geographic deletion tests and 95% **observation** coverage each have separate columns or reports.

The GP and spline integrate their latent thickness distributions over the observation hull. Positive-weight triangle quadrature tests successive refinements for all three surfaces. The global models integrate each quantile around the whole WGS84 Earth, including ash thinner than the figure shows. Polar interpolation is periodic in bearing and bilinear in log thickness. Two independent scrambled equal-area rules evaluate the posterior fields directly. Halving the mesh checks spatial resolution. [Support criteria and unsampled gaps](methods/supported-isopachs.md)

### Full-deposit panel

Evaluate the plotted median interpolant at every accepted model row, including measured values, bounds and two-sided intervals. Choose the **highest power of two in cm** whose footprint contains at least 95% of those rows, then apply a **1 cm minimum**. Membership uses the predicted thickness at the row coordinate. Disconnected patches count and holes stay outside. An observation on a contour counts inside. The caption reports the resulting count even when the floor prevents 95% coverage. The frame includes the complete plotted contour, which can extend well beyond the sites.

### Near-vent panel and area curve

All methods fitted to one input use the same square centred on its vent. Distances use distinct accepted coordinates, including bounded sites. Start with the eighth-nearest positive distance (or the farthest available). Cap it at three times the median of the nearest three positive distances, add 25% padding, and constrain the half-width to 1–100 km. Round upward at the leading decimal place. A vent-only input uses 1 km. The title gives the half-width and number of modeled rows in view. A window with no measurements shows how the model extrapolates near the source.

Observation colours use the measured thickness or reported limit, starting below 0.5 cm and doubling from 0.5 cm. Squares show two-sided ranges coloured by their lower edge. Dashed lines show conditional contours. Solid fragments, where provided, have local measurement brackets. Land and lakes use Natural Earth geography. The curve plots thickness against the square root of the **combined** area of every patch at or above that level. The area table extends to 0.5 cm for comparison across methods.

## Files behind each estimate

| File within a result directory | What it contains |
|---|---|
| `principal-volumes.csv` | Median map, integrated thickness quartiles, geographic score, coverage, numerical flags and input hash |
| `isopachs.png`, `map-display.json` | Paired regional maps on a common thickness scale and their hull limits |
| `report.json`, `thickness-bands.json` | Regional fits and fold membership, plus all three thickness integrals and quadrature refinements |
| `heldout_predictions.csv`, `selection/`, `support/` | Held-out observations, training-fold choices, unsupported holes and rejected contours |
| `<global family>/deposit.png`, `deposit.json`, `thickness-area.csv` | Full and near-vent maps, quartile integrals, convergence, contour membership by row and cumulative areas |
| `directional/azimuth/`, `spherical-mixture/` | Fitted directional and spherical posteriors |
| `directional/extent/`, `spherical-mixture/extent/` | Continuation calibration, held-out predictions and support. The spherical packet includes eight fixed priors |
| `nested-isopachs/footprints/` | Dispersal-footprint posteriors, geographic holdouts, sector contributions, support and prior sensitivities |

Earlier corridor and whole-area fits, source-sensitivity trials and sampler diagnostics retain the estimands used in their original experiments. Use the principal table for current map-based estimates. [Method definitions](methods/README.md) · [Validation studies](../comparisons/README.md)
