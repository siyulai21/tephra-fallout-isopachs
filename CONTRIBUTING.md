# Working on this repository

[Project home](README.md) · [Source map](src/README.md) · [Regional protocol](docs/methods/regional-comparison.md)

[Add an eruption](docs/NEW_ERUPTION.md) · [Add a method](docs/ADDING_METHOD.md)

The public comparison has two peer methods: GP ensemble and automatic cubic spline. Neither is a default scientific winner. New empirical models must see the same accepted inputs, outer folds, observation semantics and integration region. Method-specific tuning belongs entirely inside training folds. Never choose a method because its volume approaches a preferred published number.

Preserve raw sources and checksums; document accepted/excluded records and units. Inputs live in `data/eruptions/`, audits in `data/audits/`, and reviewed current comparisons in `eruptions/<slug>/results/`. Existing inputs must not change without a documented source correction. Ordinary runs verify audits and write a fresh output directory.

Active families live in `src/empirical_isopach/models/cubic_spline/` and `models/gaussian_process/`, registered in `models/registry.py`. Common observation and scoring tools live in `core/`. Original spline and first-extension packages are historical compatibility code. Retain old reports and their source hashes. `global/`, `regional/` and `archive/` describe historical scope/input selections, not preferred model families.

For either method, bounds are likelihood events, missing locations are not zeros, and predictions never become training observations. Do not Gaussian-moment-match the ensemble for quantiles or volume draws. Draw one GP component per whole field; draw one spline coefficient vector per whole field. Keep observation noise out of latent volumes. Conditional intervals and geographic-deletion ranges need distinct labels.

```sh
python -m pytest -q
python -m benchmarks.regional --output results/regional-methods
python -m benchmarks.support --output results/support-stress-tests
python scripts/check_repository_navigation.py
```

Inspect matched maps, all prediction failures, interval coverage and numerical refinement. Preserve unfavorable outcomes and explicitly count unestimable trials. The simulated scenarios are development evidence, not proof of universal calibration. Do not retune the algorithm from outer test outcomes without declaring a new development cycle.

Keep source/input hashes, package versions, complete candidate tables and fold membership. Large regenerated grids stay in ignored local output. Update newcomer guides, report indexes and method documentation when adding a family. New runs are requested through chat and published on GitHub; no browser workflow is required.

Support footprints and contour gates are declared development assumptions, not confidence sets. Preserve all three policies, failed contours, training-only support strata, and geographic-deletion sensitivity. Never tune closure thresholds to obtain a preferred total. A qualified perimeter does not validate its proximal interior. Derived contours are correlated summaries of one field, not extra independent observations. Keep interpolated bands and proximal/distal continuation scenarios separate; no supported band means no continuation. Fitting-domain sensitivity is distinct from changing a display or integration window.

Legacy `fit` and historical benchmark commands retain their original semantics for reproduction. Their global integrations and rectangular intervals must not be relabelled as current observation-hull volumes.
