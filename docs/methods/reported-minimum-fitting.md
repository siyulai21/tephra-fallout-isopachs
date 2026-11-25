# Fitting reported minima, upper limits and ranges

Use this rule at every distance and bearing, for each principal model, geographic holdout, prediction table, contour screen and model map. A source can describe the base of an ash bed and report only its exposed thickness. Fitting that minimum as a point lets residuals increase on either side of the reported value. This can control very thick extrapolations around a cluster of minima. It can also underestimate total fallout where the missing section was thick.

| Field report | Principal fit and held-out scoring | Model map |
|---|---|---|
| A measured thickness `t` | Point likelihood at `t`, including any documented observation error | Point symbol coloured at `t` |
| Only a lower limit `>L` or `≥L` | Same point likelihood and score as a measurement of `L` | Same point symbol and thickness colour as a measurement at `L` |
| Only an upper limit `<U` or `≤U` | Probability of thickness below `U` | Upper-limit symbol at the reporting threshold |
| A range `[L,U]` | Probability between both endpoints, including both in every fold | Interval symbol, with the lower edge used for the colour scale and both endpoints in the data |
| Occurrence without a defensible thickness | Retain its source record and evidence status | Identify separately in source or occurrence maps |

### Preparing and revising a dataset

Preserve the source row, site identifier, printed qualifier, units and original CSV checksum. The reported table keeps `Kind=lower_limit` and its positive `Limit_cm`. Register it in `reported_file`. The corresponding registered `file` is the reproducible fitting table. It preserves all row numbers and evidence IDs. A lower-only row has `Kind=measurement`, `Thickness=L`, an empty `Limit_cm`, and `Reported_lower_limit_cm=L`. The other rows keep their original fields. Store a genuine range as `Kind=interval` with `Lower_cm` and `Upper_cm`. The source audit continues to identify the reported table by its original hash.

After a source revision, regenerate the registered fitting files with `python -m benchmarks.minimum_fit_inputs --write` and verify them with `python -m benchmarks.minimum_fit_inputs`. An unregistered CSV passed to the normal fit command gets the same conversion. Archive the previous map edition and replay historical results with its own input identity.

### Changing a method or figure

Use the registered fitting table for training, tuning, validation, posterior integration and model-map overlays. Pass custom paths through `model_input`. Keep the fitting input hash in caches, figures and predictions. Source checksums alone cannot identify the observation policy. Split held-out results for directly measured thicknesses and reported minima when interpreting predictive accuracy. The density score at an assumed point and the probability of exceeding a minimum answer different questions.

Draw fitted minima with the measurement symbol. Use the original source kind on geological evidence maps whose purpose is to show what investigators reported. Use the point values for geometry, contour brackets and support calculations as well. A bracket that uses a reported minimum is conditional on accepting that value as a point. Keep its original qualifier in the ledger and map metadata. With fixed inputs, settings and seed, removing only the provenance annotation must leave all numerical fits, intervals, validation scores, contour geometry and map integrations unchanged.

Changes to dataset registration, likelihoods, validation or plot symbols must pass the four-kind regression tests and the catalogue replay. Run `python -m benchmarks.minimum_fit_results` to compare the published site values and scores. Run `python -m benchmarks.minimum_fit_release --verify` to verify each registered release against its source, geographic folds, maps and retained sampling attempts. Retain the accepted values, coordinates, duplicate decisions, sample uncertainty and exclusions from the source audit. [Input checks](../../benchmarks/minimum_fit_inputs.py) · [Regression tests](../../tests/test_minimum_fitting.py) · [Replay published site predictions and annotation checks](../../benchmarks/minimum_fit_results.py)
