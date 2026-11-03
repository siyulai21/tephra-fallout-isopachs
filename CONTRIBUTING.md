# Working on this repository

[Project home](README.md) · [Source map](src/README.md) · [Reproduction commands](benchmarks/README.md)

Keep scientific changes distinct from reporting changes. `model.py`, `validation.py`, `ensemble.py`, `distributions.py` and observation semantics define the fit and predictive distribution. The 0.3 reporting release preserved the selected-model fit; the 0.4 ensemble is opt-in and requires independent evaluation. Never summarize its field as a single Gaussian for quantiles, thickness expectations or volume sampling.

Reporting belongs in `contours.py`, `volume.py`, `figures.py` and `publication.py`. Never use a display boundary or hand-closed curve to constrain the fit. Keep unresolved results visible.

```sh
python -m pip install '.[test]'
python -m pytest -q
python -m benchmarks.empirical --output results/empirical --repeats 10
python -m benchmarks.minoan --output results/review-minoan
```

Inspect maps, compare model coefficients when fitting should be unchanged, and inspect area/volume refinement histories. Run the full synthetic benchmark when changing the scientific model or validation design.

Keep raw input bytes unchanged unless correcting a documented source error. Generate into a new directory before replacing a published report. `reports/` is a reviewable build; `results/` holds scratch runs and large arrays. Do not fabricate provenance or claim total-volume validation.

Original `src.*` imports and the separate spline GUI are retained for compatibility. The browser upload server has been removed. New analyses are requested in chat and published as GitHub-readable Markdown reports with figures, tables and numerical records.


For an uploaded eruption, preserve the raw source and its checksum, document accepted/excluded records and units, and keep sensitivity overrides separate from the main input. Save accepted CSVs in `data/eruptions/`, audits in `data/audits/<eruption>/`, and reviewable results in `reports/<eruption>/`. Do not infer measurement error from a range without a defensible observation model. Record commands, package versions and code hashes; leave large prediction arrays in `results/`.

The audited YTT comparison is reproduced with `python -m benchmarks.ytt --output results/ytt`. Keep the selected-GP default unless matched geographic holdouts show a defensible improvement in proper predictive scores, with coverage and point errors also examined. Publish unfavorable results and site-level failures. Four folds on one deposit do not establish general superiority.

The 0.6 likelihood adds documented lower bounds using log survival probabilities. Keep likelihood derivatives, model selection, stacking, calibration and outer evaluation consistent about censor direction. Coverage and RMSE use exact measurements only. For Minoan, combine fallout at the observation level; do not fit separate fans and sum marginal interval endpoints.

Version 0.7 makes `global_fallout.selected` the primary volume. Preserve exact independence from `--bounds`, display extent and contour levels. The global ellipse must match the WGS84 LAEA image; do not use an infinite plane, a spherical circle with ellipsoidal coordinates, or a thickness cutoff. Retain analytic/independent-geometric tests, failure flags and mixture semantics. Regional simulation intervals cannot be reused globally. Run `python -m benchmarks.global_fallout --output results/global-fallout` for all six eruptions and known-truth trials.

## Keep the repository understandable

Every current eruption calculation uses a GP model. Say “selected GP” or “GP ensemble” in human-facing text; do not use “original” or “established” to imply a cubic-spline result. The retained original spline lives in `src/spline_isopach/`; the trend-regularized spline experiment lives in `src/research/` with a compatibility import at its previous path.

Each eruption's root README is its current entry point. Put complete-Earth calculations under `global/` and earlier finite-region artifacts under `regional/`. Preserve superseded input versions explicitly in `archive/`. Do not overwrite numerical files or source hashes during navigation changes. Recorded paths inside a historical JSON describe its original run.

Update the root README, report catalogue and relevant folder indexes when adding an eruption or method. Link scientific claims to their actual calculation and input selection. Dated benchmarks belong in `docs/validation/`; method derivations belong in `docs/methods/`.

```sh
python scripts/check_repository_navigation.py
python -m pytest -q
```

The navigation check validates local Markdown targets and images, not external websites or heading fragments. Keep dataset/source paths stable; publication paths can move with their links and a documented mapping. Check that existing numerical artifacts remain byte-identical when only their location changes.
