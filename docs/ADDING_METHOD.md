# Add an empirical model family

[Method catalogue](../methods/README.md) · [Source map](../src/README.md) · [Contributing](../CONTRIBUTING.md)

A new family joins the same eruption workflow as a peer. It must not silently change the accepted data, the target region or another method's results.

The registry contract below applies to **regional** model families. The [global mass model](methods/probabilistic-global.md) has a separate joint-posterior runner because its whole-Earth volume parameter and uncertainty do not fit the fixed-parameter regional interface. A global extension must preserve the same accepted observations, bounds, outer folds, buffer and heldout scoring; it must also supply complete-field draws, exact or independently verified mass normalization, explicit prior sensitivity and numerical sampling diagnostics. It must never compare its global total directly with a regional partial volume as if they estimated the same quantity.

## Implementation boundary

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

## Validation and publication

Test the new numerical solver against an independent reference or known case, test observation/bound semantics and leakage prevention, and check correlated draws and integral refinement. Run the matched real-data and known-truth comparisons without selecting a winner in advance. Keep non-convergence, insufficient support and poor coverage visible.

Add a method card under `methods/`, its mathematical protocol under `docs/methods/`, and reviewed comparative evidence under `comparisons/`. When the active protocol changes, rerun before describing a comparison as current. Reports record their own method identities and labels so older artifacts remain readable after the registry changes.

The registry is a navigation and dispatch mechanism, not evidence that a newly registered method is scientifically validated. New predictive distributions or new data requirements may require extending the shared contract deliberately.
