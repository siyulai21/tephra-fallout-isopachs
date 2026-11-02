# Working on this repository

Keep scientific changes distinct from reporting changes. `model.py`, `validation.py`, `ensemble.py`, `distributions.py` and observation semantics define the fit and predictive distribution. The 0.3 reporting release preserved the selected-model fit; the 0.4 ensemble is opt-in and requires independent evaluation. Never summarize its field as a single Gaussian for quantiles, thickness expectations or volume sampling.

Reporting belongs in `contours.py`, `volume.py`, `figures.py` and `publication.py`. Never use a display boundary or hand-closed curve to constrain the fit. Keep unresolved results visible.

```sh
python -m pip install '.[test]'
python -m pytest -q
python -m benchmarks.empirical --output results/empirical --repeats 10
python -m src.probabilistic_isopach atlas --output results/review-atlas
```

Inspect maps, compare model coefficients when fitting should be unchanged, and inspect area/volume refinement histories. Run the full synthetic benchmark when changing the scientific model or validation design.

Keep raw input bytes unchanged unless correcting a documented source error. Generate into a new directory before replacing a published atlas. `docs/atlas/` is a reviewable build; `results/` holds scratch runs and large arrays. Do not fabricate provenance or claim total-volume validation.

Original `src.*` imports and the separate spline GUI are retained for compatibility. The upload workspace runs analyses in separate subprocesses and isolated output directories.
