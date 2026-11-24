# Working on this repository

[Project home](README.md) · [Source map](src/README.md) · [Regional protocol](docs/methods/regional-comparison.md)

[Add an eruption](docs/NEW_ERUPTION.md) · [Add a method](docs/ADDING_METHOD.md)

The workflow has two regional peers (GP ensemble and automatic cubic spline) and two conditional global models, directional continuation and the spherical mixture. Neither is a default scientific winner. New models must see the same accepted inputs, outer folds and observation semantics. Regional peers share their integration region; the global models use the physical Earth. Method-specific tuning belongs entirely inside training folds. Never choose a method because its volume approaches a preferred published number.

Preserve raw sources, checksums, units and site-level decisions. Inputs live in `data/eruptions/`, audits in `data/audits/`, and reviewed comparisons in `eruptions/<slug>/results/`. An input revision can correct a source error or incorporate a documented reinterpretation. Trace sites without measured thickness to the primary evidence and record the derivation of each principal bound or the grounds for omission. Keep the earlier input and identify its revision. Ordinary runs verify audits and write a fresh output directory. [Site decision protocol](docs/methods/sites-without-measured-thickness.md)

Active families live under `src/empirical_isopach/models/`: regional `cubic_spline/` and `gaussian_process/` use `registry.py`; global `directional/`, `mass_mixture/` and `nested/` have separate joint-posterior runners. Common observation and scoring tools live in `core/`. Original spline and first-extension packages are historical compatibility code. Preserve source hashes and identify preceding results by immutable Git commit. Keep required references beside their study; unused archived report copies belong in Git history. Case outputs put the global families in `results/directional/`, `results/spherical-mixture/` and `results/nested-isopachs/`. Use the dispersal-footprint posterior for the principal nested map and volume and publish the separate sensitivity in `results/nested-whole-area/`. Link to the three principal reports directly from the case home.

For every family, bounds are likelihood events, missing locations are not zeros, and predictions never become training observations. Do not Gaussian-moment-match the ensemble for quantiles or volume draws. Draw one GP component per whole field; draw one spline coefficient vector per whole field. Keep observation noise out of latent volumes. Integrate the median and both quartile thickness maps on the same domain. Record the held-out 95% predictive coverage and geographic-deletion sensitivity separately.

```sh
python -m pytest -q
python -m benchmarks.regional --output results/regional-methods
python -m benchmarks.support --output results/support-stress-tests
python scripts/check_repository_navigation.py
```

Inspect matched maps, all prediction failures, interval coverage and numerical refinement. Preserve unfavorable outcomes and explicitly count unestimable trials. The simulated scenarios are development evidence, not proof of universal calibration. Do not retune the algorithm from outer test outcomes without declaring a new development cycle.

Keep source/input hashes, package versions, complete candidate tables and fold membership. Large regenerated grids stay in ignored local output. Update newcomer guides, report indexes and method documentation when adding a family. New runs are requested through chat and published on GitHub; no browser workflow is required.

Support footprints and contour gates are declared development assumptions, not confidence sets. Preserve all three policies, failed contours, training-only support strata, and geographic-deletion sensitivity. Never tune closure thresholds to obtain a preferred total. A qualified perimeter does not validate its proximal interior. Derived contours are correlated summaries of one field, not extra independent observations. No supported band means no *contour-band continuation*. At the user's explicit request, this is no longer a veto on conditional global inference: run both [global families](docs/methods/README.md), their fixed sensitivities and matching geographic holdouts. Fitting-domain sensitivity is distinct from changing a display or integration window.

The spherical-mixture companion lives in `models/mass_mixture/`; it is separate from the regional registry because it represents a joint non-Gaussian parameter posterior rather than the regional families' fixed-parameter uncertainty contract. It must retain observation semantics, comparable heldout likelihood scores and all numerical diagnostics. Global totals are normalized on WGS84, with no arbitrary integration boundary. Publish baseline full-field samples and sensitivity volume draws; retain signatures for all numerical attempts. For directional changes, replay the principal fit with `python -m benchmarks.verify_azimuth_continuation`. Check its fixed-direction components with `python -m benchmarks.verify_directional_ensemble` and its 13-policy reference with `python -m benchmarks.verify_directional`. [Directional validation](comparisons/directional/README.md) Add `python -m benchmarks.mass_mixture --output results/global-stress` to spherical-mixture numerical validation.

Legacy `fit` and historical benchmark commands retain their original semantics for reproduction. Their global integrations and rectangular intervals must not be relabelled as current observation-hull volumes.

Changes to nested isopachs require matched geographic validation, analytic and geographic integral checks, and known-volume and contour challenges. The paired directional and spherical sensitivity studies use the same accepted sites and training-fold posteriors. [Nested validation and estimates](comparisons/nested-isopachs/README.md)
