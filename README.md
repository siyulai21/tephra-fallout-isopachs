# tephra-fallout-isopachs

Reconstruct tephra thickness and isopachs with an explicit distinction between measurements, non-detections, and assumptions. The reproducible workflow needs no hand-placed pseudo-points. Existing datasets and the original cubic spline interface are retained.

**Research status:** working, tested implementations, not a validated replacement for geological interpretation. Conditional intervals omit uncertainty in fitted parameters and model selection. Aso-4 and the synthetic missing-sector experiment fail important uncertainty checks. Their intervals must not be presented as calibrated eruption-volume intervals.

## Statistical treatments

1. **Predict unobserved thickness instead of inventing observations.** A directional thinning trend plus a Matérn Gaussian process gives a joint distribution in gaps. Exponential versus power thinning and smoothing are selected using deterministic spatial validation on real measurements. Legacy pseudo rows cannot influence fitting, selection, or the integration domain.
2. **Retain cubic splines with a normalized spatial prior.** An area-averaged penalty replaces arbitrary pseudo-point likelihoods. Total prior information stays fixed when its numerical grid is refined; validation chooses its strength, including zero. This produces point predictions, not posterior intervals.
3. **Represent genuine constraints statistically.** Documented non-detections use a censored likelihood. Independently justified external evidence can use a full correlated error covariance, so repeating the same statement does not make it independent evidence.

Read [the mathematics and assumptions](docs/METHODS.md) and [the executed validation](docs/VALIDATION.md). These combine established statistical ideas into a project-specific workflow; scientific novelty has not been established by a systematic literature review.

## Run an analysis

Python 3.10 or later:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
python -m src.probabilistic_isopach examples/Data/Mazama_tephra.csv \
  --output results/mazama --nested-cv
```

After installation, `tephra-isopachs` is equivalent to `python -m src.probabilistic_isopach`. `requirements-tested.txt` records the exact benchmark environment; it is an environment snapshot, not a cross-platform lock file.

Each run writes:

- `isopachs.png`: median contours, conditional uncertainty, and alternative-tail sensitivity.
- `predictions.csv`: geographic/projected coordinates, thickness summaries, conditional latent-surface intervals, and distance to observations.
- `report.json`: input/code hashes, package versions, model choices, fold assignments, numerical integration check, warnings, and domain sensitivity.
- `virtual_predictions.npz`: mean **and full covariance** of predicted log thickness. These are predictions, never extra independent training data.

An existing nonempty output directory is rejected. `--nested-cv` estimates performance with separate outer spatial folds. Without it, CV values are selection diagnostics, not independent performance estimates. `--buffer-km` adds geographic separation; insufficient training data fail visibly.

Volume is integrated over a finite rectangle, with a grid-refinement check. By default it contains the vent and all real observations with 25% padding. `--bounds XMIN YMIN XMAX YMAX` overrides it in vent-centred equal-area kilometres. Enlarged-domain results are always reported. These are **not total erupted volumes**.

## Input data

Existing CSVs work unchanged. Required columns: `Lon`, `Lat`, `Thickness`, `Vent Lon`, `Vent Lat`. Coordinates are decimal degrees and thickness is centimetres. Vent coordinates may occur on any row but must be consistent. Optional `Pseudo` accepts blank/No/False/0 for observations and Yes/True/1 for excluded invented values.

| Lon | Lat | Thickness | Pseudo | Vent Lon | Vent Lat | Kind | Limit_cm | LogSD |
|---:|---:|---:|:---|---:|---:|:---|---:|---:|
| -122.37 | 43.05 | 850 | | -122.1 | 42.95 | measurement | | 0.10 |
| -124.19 | 42.03 | | | | | upper_limit | 0.5 | 0.20 |
| -129.96 | 55.43 | 0.01 | Yes | | | | | |

The upper-limit row is illustrative, not evidence about that location. Use upper limits only when actual detection and preservation information justifies the bound. `LogSD` is known additional standard deviation of **natural-log thickness**, not centimetres. A fitted residual nugget is added in quadrature; missing `LogSD` does not imply zero total noise.

Zeros, missing real thicknesses, unsupported trace labels, invalid coordinates, and ambiguous flags raise errors. “Not sampled”, “eroded”, and “not reported” are not zero original deposition. Old positive values substituted for traces cannot be detected automatically and need source review. Unnamed extra columns are ignored and listed in the report.

## Spline prior API

```python
from src.probabilistic_isopach import read_observations
from src.probabilistic_isopach.regularized_spline import select_spline_prior

data, metadata = read_observations('examples/Data/Mazama_tephra.csv')
model, selection = select_spline_prior(data)
predicted_log_thickness = model.predict_log(data.xy)
```

This estimator supports positive measurements and refuses predictions outside its rectangle. Its CV uses a known survey rectangle from all real sampling coordinates; thickness-dependent fitting uses only training folds. Scores select strength and are not nested performance estimates. Inspect boundary optima in the finite strength search.

## Validation and original spline

```sh
python -m pytest -q
python -m benchmarks.run --output results/benchmark --repeats 10
```

The benchmark runs nested spatial validation on four datasets, two fixed baselines, and 30 synthetic experiments. The spline baseline uses the **corrected** original solver, historical tuning defaults, and no pseudo-points. This is not a reproduction of published methods or a comparison against fully tuned alternatives.

For the legacy interactive interface, install `'.[legacy-map]'` and run `python -m src.spline_isopach`. It retains manual pseudo-points. The numerical solver now includes every basis cross term, uses correct band storage, solves the complete final grid, fixes the aspect ratio, preserves sub-centimetre values, and rejects open contours when calculating area. These corrections change historical outputs. The optional Basemap GUI was not exercised; its numerical core and contour-area calculations are tested.
