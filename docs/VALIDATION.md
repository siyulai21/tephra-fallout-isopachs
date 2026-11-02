# Executed validation

This is a diagnostic research benchmark, executed on 1 November 2026. It tests reproducibility and exposes limitations; it does not validate total erupted volumes. Machine-readable results, source/input hashes, and environment details are in [validation.json](validation.json). The numerical test suite passed **25 tests**.

## Real datasets: nested geographic holdouts

Four outer spatial folds, three inner selection folds, no buffer. All legacy pseudo rows are excluded. Near-colocated observations across fold boundaries can still create spatial dependence; a buffered sensitivity analysis has not been run. Metrics are on natural-log thickness. Nominal 95% coverage concerns withheld observations including their residual nugget, not coverage of true latent thickness or total volume.

| Dataset | Unflagged observations | Excluded pseudo rows | GP log RMSE | Fixed trend RMSE | Corrected spline RMSE | GP nominal-95% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Mazama | 45 | 7 | 0.907 | 0.872 | 1.414 | 93.3% |
| Tambora | 26 | 2 | 1.484 | 1.545 | 2.170 | 92.3% |
| Aso-4 | 87 | 6 | 1.137 | 1.184 | 1.262 | 79.3% |
| B-Tm | 84 | 7 | 0.885 | 0.906 | 1.241 | 97.6% |

The GP workflow has lower error than the corrected spline at its original tuning defaults on all four datasets. It does **not** dominate the simpler exponential trend: that trend performs slightly better on Mazama. The spline and trend baselines are fixed, not independently tuned, so this is not evidence of universal superiority. The spline baseline clips outside its training rectangle, retaining the historical prediction behavior; the new GP explicitly extrapolates.

**Aso-4 fails the uncertainty check.** Only 69 of 87 held-out observations (79.3%) lie inside its nominal 95% intervals. Its fitted power mean touches the decay constraint and roughly 67% of expected volume lies outside the observation hull. Increasing domain widths by a factor of two increases its conditional mean volume from about 647 to 1,317 km³; the alternative thinning family reaches about 2,968 km³ in that larger rectangle. These are demonstrations of unresolved extrapolation, not defensible new eruption-volume estimates.

B-Tm also has material domain sensitivity: expected finite-domain volume increases from about 40 to 54 km³ on doubling rectangle widths. No automatic total-volume conclusion is issued.

The initial Tambora 24-to-48-cell grid check changed the mean integral by 2.3%, triggering a refinement warning. A separate 48-to-96-cell check gave a relative change of 0.29%. The latter checks only the expected integral; Monte Carlo quantile convergence was not established.

All dataset thicknesses were accepted according to the existing CSV flags. This benchmark did not recover or independently audit each underlying field source. A positive value historically substituted for a trace or non-detection would still be treated as a measurement until its provenance is corrected.

## Spline prior: strength selection diagnostics

The table compares the no-prior control and the selected prior on the same fixed-survey-domain folds. These are selection scores, **not independent or nested performance estimates**, and should not be compared directly with the nested table above.

| Dataset | Selected total strength | No-prior log RMSE | Selected log RMSE | Boundary optimum? |
|---|---:|---:|---:|:---|
| Mazama | 100 | 1.369 | 0.972 | Yes |
| Tambora | 1 | 2.103 | 1.325 | No |
| Aso-4 | 100 | 1.369 | 1.167 | Yes |
| B-Tm | 100 | 1.660 | 1.175 | Yes |

Three optima are at the maximum tested strength. The finite grid therefore does not establish an optimum for those datasets; an expanded search and outer evaluation are needed before comparative performance claims. The reliable mathematical result is that grid replication does not strengthen the prior: this is tested directly, separately from whether a particular trend assumption is scientifically appropriate.

## Synthetic stress tests

Ten seeds for each of three scenarios, 60 noisy training measurements and 400 fresh evaluation locations per run. Observation noise SD is 0.25 in natural-log units. The secondary-lobe case adds a localized thickness enhancement; the missing-sector case withholds the quadrant containing that enhancement. These simulations are not generated from the fitted GP. Volume errors use the same random evaluation grid for truth and prediction, so they are diagnostic finite-domain errors, not independently converged volume estimates.

| Scenario | Mean latent log RMSE | Latent nominal-95% coverage | Observation nominal-95% coverage | Mean volume relative error |
|---|---:|---:|---:|---:|
| smooth | 0.092 | 98.2% | 90.1% | -0.2% |
| secondary_lobe | 0.273 | 95.5% | 98.6% | +5.3% |
| missing_sector | 0.424 | 83.5% | 76.9% | -23.9% |

The missing-sector experiment underestimates finite-domain volume by about 24% on average and covers only 77% of new noisy observations. The workflow cannot infer a completely unobserved lobe from these observations. Smooth-case observation coverage is also only about 90%; omitted parameter uncertainty and conditional modelling matter even in a comparatively simple case. There is no claim of universally calibrated uncertainty.

## Numerical and statistical checks

The 25 automated tests cover exact GP conditioning against independent dense equations; censored-likelihood derivatives and weak/strong limits; invariance to arbitrary legacy pseudo values/replication; correlated evidence duplication; deterministic colocated spatial folds; absence of held-out response leakage in fits; nested row assignment; directional decay; correlated volume sampling and units; complete spline Gram matrices and normal equations; the direct and fallback solvers; sub-centimetre functional predictions; signed contour holes and refusal to close open contours; and normalized prior duplication/refinement/zero-strength behavior.

The default CLI ran end to end for all four datasets and produced maps, machine-readable grids, joint virtual distributions, and complete reports. The optional Basemap GUI was not tested. No untouched external eruption dataset, physical-simulator validation, parameter-posterior inference, or repeated-buffer study has been completed.

Reproduce the main benchmark with:

```sh
python -m pytest -q
python -m benchmarks.run --output results/benchmark --repeats 10
```

Full run artifacts were retained locally under `results/validated/`; large generated files are excluded from Git. The compact results here intentionally preserve unfavorable findings alongside improvements.
