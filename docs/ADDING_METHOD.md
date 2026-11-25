# Add an empirical model family

[Method catalogue](methods/README.md) · [Source map](../src/README.md) · [Contributing](../CONTRIBUTING.md)

A new family joins the same eruption workflow as a peer. It must not silently change the accepted data, the target region or another method's results.

The registry contract below applies to **regional** model families. The [directional ensemble](methods/directional-multiscale.md) and [spherical mixture](methods/spherical-bearings.md) have separate joint-posterior runners because their complete-field parameter uncertainty does not fit the fixed-parameter regional interface. A global extension must preserve the same accepted observations, bounds, outer folds, buffer and heldout scoring; it must also supply complete-field draws, exact or independently verified mass normalization, explicit prior sensitivity and numerical sampling diagnostics. It must never compare its global total directly with a regional partial volume as if they estimated the same quantity.


Every new family must fit a lower-only site as a point at its reported minimum, fit an upper limit with a cumulative probability, and fit a range with the probability between its two endpoints. Use the matching fit identity in validation, caches and figures. [Observation and map contract](methods/reported-minimum-fitting.md)

## Regional implementation boundary

Put the implementation in `src/empirical_isopach/models/<family>/`. Add an adapter and one entry in [registry.py](../src/empirical_isopach/models/registry.py) with a stable identifier, label, description, fitting function and optional diagnostic callback. The registry order only controls presentation. The comparison runner, geographic evaluation, report panels and benchmark summaries iterate registered methods; there is no preferred-method field.

`fit(data, bounds, folds, buffer_km)` returns `(model, selection)`. The observations have already been standardized into projected kilometres, natural-log thickness and explicit exact/upper/lower semantics. The fitting bounds use survey coordinates only. Every response-dependent choice belongs inside the supplied training set and its inner geographic folds.

## Statistical and output contract

| Contract | Required behavior |
|---|---|
| Observations | Preserve exact values, upper/lower likelihood events, known errors and row identities; never add predictions as observations |
| Predictive distribution | Supply `predict_log` with latent and observation modes for a conditional Gaussian field, or `distribution` implementing the marginal probability interface used in [core/distributions.py](../src/empirical_isopach/core/distributions.py) |
| Correlated volume uncertainty | The current draw engine supports joint Gaussian covariance, whole-field Gaussian mixtures, and cubic coefficient draws. A different family must extend and test this explicitly; independent pointwise sampling is not a valid substitute |
| Selection record | Include `candidates`, with parameters, fold scores and failures; record the selected model/weights and any search limits. An untuned model should explicitly record its fixed candidate |
| Description | `model.describe()` returns JSON-serializable settings sufficient to identify the fitted method |
| Diagnostics | An optional callback receives the fitted model, data, bounds, region and result; returns extra checks and report notes without modifying observations |
| Uncertainty labels | State precisely what is conditional and which uncertainties are omitted; do not invent covariance or claim calibrated coverage |

Full outer geographic holdouts rerun the entire tuning procedure. Numerical integration uses the same observation hull and units for every family. Each method keeps its own surface and volume; adding a method does not create an ensemble across families.

## Global integration boundary

Add a global family under `models/<family>/` with a complete-field posterior, its own reporting runner and fixed sensitivity protocol. Wire it into the ordinary `compare` command, the method listing and a clearly named peer output directory. Rebuild all response-dependent geometry within training folds. Retain the shared support screen, numerical attempts, whole-field mass accounting and the matched geographic evaluation. Never turn another fitted surface or derived contour into independent training evidence. Publish a method card, implementation map, assumptions and known-volume challenges; do not describe a new family as validated merely because it runs.

## Visual and geological review of isopachs

Apply the following criteria together when revising an existing method or proposing a new one. Compare the candidate and the current fits on the same accepted sites, with matching contour levels, map extents and scales. Inspect the rendered median-thickness field used for the volume estimate. For a regional method, the widest view is its observation hull. For a global method, inspect the full deposit and distinguish measured brackets from conditional extrapolation.

Start with the full map and locate the vent and innermost contours on it. Follow the same fan and thickness levels through intermediate distances and into the near-vent view, then return to the full picture. A plausible close-up can still join the distal fan at an implausible angle. If a published panel cannot resolve that transition, inspect a closer view of the same fitted field. [Map and contour conventions](OUTPUTS.md#read-the-map-and-its-volume)

| Criterion | What to look for |
|---|---|
| Contour smoothness | Trace one level for jagged edges, oscillations or sharp corners. Check whether they persist when the same field is sampled on a finer grid. |
| Curvature continuity | Follow bends along the line. A smooth-looking line can still switch suddenly from a broad arc to a tight turn without a corresponding change in nearby thicknesses. |
| Inter-contour coherence | Compare adjacent levels for a lobe that appears, disappears or changes shape abruptly. Look for sites that explain the change. |
| Contour spacing | Along and across each fan, look for successive doubling levels that suddenly bunch together or spread apart. Check the implied thinning rate against the local measurements. |
| Proximal–distal continuity | Trace the thick inner contours into the wider deposit. Check for artificial bullseyes, visible seams and inner contours geometrically disconnected from the distal pattern. |
| Dispersal-axis continuity | Follow each fan’s direction and lateral position across thickness levels. Investigate abrupt rotations or offsets, allowing for a second fan where the observations support it. |
| Topology | Inspect holes, isolated patches, narrow protrusions and secondary maxima. Distinguish a low-thickness area predicted by the fit from a gap in observational support. Preserve support gaps on the map. |
| Source geometry | Compare near-vent gradients and the position of each thickness maximum with the mapped source and proximal sites. Check whether the evidence supports an offset maximum. |
| Geographic extrapolation | Trace lobes or broad radial spreading beyond the last constraining sites. Use the support and sensitivity results to identify how much of that reach depends on the model. |
| Upwind deposition | Where dispersal direction is documented, compare upwind reach and predicted thickness with observations in that sector. If the direction or upwind thickness is poorly constrained, record that uncertainty. Unsampled ground supplies no zero. |
| Geological flexibility | Check whether near-perfect ellipses or rigid symmetry erase supported asymmetry, overlapping fallout fans or a shift in dispersal direction between phases. |
| Observation agreement | Inspect signed errors at measured sites and predictions against the accepted limits and ranges under the registered fitting policy. Look for repeated overprediction or underprediction by distance and sector, including in geographic holdouts. |

In the existing method comparison or pull request, record material findings with the eruption, map scale, contour levels, sector, and the site IDs or support diagnostics that bear on them. For example, if the 8 cm contour turns away from the 16 cm contour, identify which sites bracket the turn and whether it persists when that geographic group is withheld. Record where the evidence supports a feature and where it leaves competing shapes plausible. Weigh these findings alongside held-out predictions, uncertainty and numerical checks. [Contour support protocol](methods/supported-isopachs.md)

Trace a suspected artifact to its cause. If it disappears on a finer evaluation grid, correct the rendering and replay the same field. If it persists in the fitted thicknesses, investigate the model assumption responsible, such as a forced common axis or an abrupt change in thinning rate. Test the revision on matched data and folds, and check its effect on contours, site predictions and the integrated volume. Preserve genuine geological complexity. Smoothing plotted lines, imposing symmetry or hand-editing a lobe would break the connection between the displayed isopachs and the field being evaluated.

## Validation and publication

Test the new numerical solver against an independent reference or known case, test observation/bound semantics and leakage prevention, and check correlated draws and integral refinement. Run the matched real-data and known-truth comparisons without selecting a winner in advance. Keep non-convergence, insufficient support and poor coverage visible.

Add the method to the [method guide](methods/README.md), place its equations alongside the other protocols and publish reviewed evidence under `comparisons/`. When the active protocol changes, rerun before describing a comparison as current. Reports record their own method identities and labels so older artifacts remain readable after the registry changes.

The registry is a navigation and dispatch mechanism, not evidence that a newly registered method is scientifically validated. New predictive distributions or new data requirements may require extending the shared contract deliberately.

For a method that extrapolates across a sampling gap, test archived no-bed readings at their own coordinates. Show the inferred response to an intact visible bed and its sensitivity to loss or mixing. Use the same audited quantitative input for the geographic benchmark. [Archive observation and continuation protocol](methods/archive-evidence.md)

Bayesian nested isopachs use joint geographic draws, matched withholding and direct map integration on the physical Earth. The model's analytic total for each complete draw supplies an independent check. [Principal implementation and scientific limits](methods/nested-footprints.md)

Report one point estimate of bulk fallout for each fitted surface. Integrate its pointwise median thickness on the stated domain. Integrate the pointwise 25th and 75th percentiles of latent thickness over that same domain and verify both endpoints. If a sensitivity lacks an integrated median field, report its uncertainty and predictions only.
