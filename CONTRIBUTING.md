# Working on this repository

[Project home](README.md) · [Source map](src/README.md) · [Regional protocol](docs/methods/regional-comparison.md)

[Add an eruption](docs/NEW_ERUPTION.md) · [Add a method](docs/ADDING_METHOD.md)

The workflow has two regional peers (GP ensemble and automatic cubic spline) and a separate conditional global posterior companion. Neither is a default scientific winner. New models must see the same accepted inputs, outer folds and observation semantics. Regional peers share their integration region; the global companion uses the physical Earth. Method-specific tuning belongs entirely inside training folds. Never choose a method because its volume approaches a preferred published number.

Preserve raw sources and checksums; document accepted/excluded records and units. Inputs live in `data/eruptions/`, audits in `data/audits/`, and reviewed current comparisons in `eruptions/<slug>/results/`. Existing inputs must not change without a documented source correction. Ordinary runs verify audits and write a fresh output directory.

Active families live in `src/empirical_isopach/models/cubic_spline/` and `models/gaussian_process/`, registered in `models/registry.py`. Common observation and scoring tools live in `core/`. Original spline and first-extension packages are historical compatibility code. Preserve source hashes and identify preceding results by immutable Git commit. Keep required references beside their study; unused archived report copies belong in Git history. `results/global/` is the current conditional global companion, not a historical-method label.

For either method, bounds are likelihood events, missing locations are not zeros, and predictions never become training observations. Do not Gaussian-moment-match the ensemble for quantiles or volume draws. Draw one GP component per whole field; draw one spline coefficient vector per whole field. Keep observation noise out of latent volumes. Conditional intervals and geographic-deletion ranges need distinct labels.

```sh
python -m pytest -q
python -m benchmarks.regional --output results/regional-methods
python -m benchmarks.support --output results/support-stress-tests
python scripts/check_repository_navigation.py
```

Inspect matched maps, all prediction failures, interval coverage and numerical refinement. Preserve unfavorable outcomes and explicitly count unestimable trials. The simulated scenarios are development evidence, not proof of universal calibration. Do not retune the algorithm from outer test outcomes without declaring a new development cycle.

Keep source/input hashes, package versions, complete candidate tables and fold membership. Large regenerated grids stay in ignored local output. Update newcomer guides, report indexes and method documentation when adding a family. New runs are requested through chat and published on GitHub; no browser workflow is required.

Support footprints and contour gates are declared development assumptions, not confidence sets. Preserve all three policies, failed contours, training-only support strata, and geographic-deletion sensitivity. Never tune closure thresholds to obtain a preferred total. A qualified perimeter does not validate its proximal interior. Derived contours are correlated summaries of one field, not extra independent observations. No supported band means no *contour-band continuation*. At the user's explicit request, this is no longer a veto on conditional global inference: run the separate [mass-normalized posterior model](docs/methods/probabilistic-global.md), its fixed sensitivities and matching geographic holdouts. Fitting-domain sensitivity is distinct from changing a display or integration window.

The global companion lives in `models/mass_mixture/`; it is separate from the regional registry because it represents a joint non-Gaussian parameter posterior rather than the regional families' fixed-parameter uncertainty contract. It must retain observation semantics, comparable heldout likelihood scores and all numerical diagnostics. Global totals are normalized on WGS84, with no arbitrary integration boundary. Publish baseline full-field samples and sensitivity volume draws; retain signatures for all numerical attempts. Add `python -m benchmarks.mass_mixture --output results/global-stress` to numerical method validation.

Legacy `fit` and historical benchmark commands retain their original semantics for reproduction. Their global integrations and rectangular intervals must not be relabelled as current observation-hull volumes.
