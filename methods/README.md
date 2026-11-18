# Four empirical families, two volume questions

[Project home](../README.md) · [Eruption results](../eruptions/README.md) · [Validation evidence](../comparisons/README.md) · [Add a family](../docs/ADDING_METHOD.md)

Every family starts from the **same accepted measurements, bounds and source location**. Each fits its own thickness field and calculates its own volume. No family is the preferred answer, and their volumes are never averaged together.

| Scope | Family | How it constructs thickness | Volume reported | Guide |
|---|---|---|---|---|
| Regional | **GP ensemble** (`stacked`) | Fitted thinning trends plus correlated GP residuals; predictive stacking combines candidate fields | Common observation hull | [GP ensemble](gaussian-process.md) |
| Regional | **Automatic cubic spline** (`spline`) | Cubic B-spline basis with normalized derivative and fitted-trend penalties; geographic training folds choose knots and regularization | The same observation hull | [Cubic spline](cubic-spline.md) |
| Global | **Directional continuation** (`directional`) | Latent log thickness is linear within measurement triangles; a positive directional thinning law continues through gaps and beyond the sampled ground | Whole Earth, partitioned by compass direction and sampling support | [Directional continuation](directional-fallout.md) |
| Global | **Spherical mixture** (`spherical-mixture` output folder) | A positive mixture of fallout fans with normalized mass and a joint posterior for volume and geometry | Whole Earth, with explicit prior sensitivities | [Spherical mixture](spherical-mixture.md) |

## How the newest method relates to the others

The directional family was **added alongside the spherical mixture in version 0.12**. It does not consume GP predictions, spline contours or the spherical posterior as observations. Its isopach fragments are summaries of its own fitted field, not extra independent data. It uses neither a GP residual covariance nor the cubic-spline solver.

It reuses shared observation parsing, the local support mesh, geographic folds, probability tools and some spherical geometry/sampling utilities. Sharing those utilities keeps the comparison consistent; it does not make one model a stage of another. [Implementation map](../src/empirical_isopach/models/README.md).

The main scientific change concerns **unsampled directions**. The spherical mixture fits whole-deposit fan shapes. Directional continuation anchors local interpolation to measured triangles and limits how far a measured bearing can influence neighbouring directions. Empty bearings still receive positive fallout. The fixed missing-fan and proximal-cap sensitivities expose alternative deposits that the measurements cannot rule out. [Equations and assumptions](../docs/methods/directional-continuation.md).

The newest family is not a demonstrated replacement: its geographic score is lower in 2/6 eruptions, fresh known-volume intervals cover truth in 4/8 numerically passing trials, and all four invisible-lobe trials miss the larger, observationally indistinguishable deposit. Tambora retains a sampling warning. These are the published development findings, not universal performance rates. [Matched evidence and limitations](../comparisons/directional/README.md).

## Shared rules and different uncertainty

All four use accepted observations and documented likelihood bounds. None requires hand-placed pseudo-observations. Integration nodes, spline penalty points and directional transition vertices supply numerical geometry, not evidence. Automatic fitting removes per-eruption manual tuning; fixed priors, geometry rules and search boundaries remain assumptions.

GP and spline use identical regional hulls and nested geographic selection. Their conditional volume draws omit some fitted-parameter and selection uncertainty. The two global families sample joint parameter/field posteriors under their own priors. Keep each conditional interval separate from geographic-deletion ranges and sensitivity envelopes. A regional volume and a whole-Earth volume answer different questions.

The [support audit](../docs/methods/supported-isopachs.md) applies to every family. It preserves holes and rejected contours; a failure does not veto a conditional global calculation. Held-out prediction, numerical convergence and identification of unseen global mass are different checks.

## Running and extending the catalogue

`tephra-isopachs compare` runs all four families. The [regional registry](../src/empirical_isopach/models/registry.py) lists GP and spline; the two global families use separate joint-posterior runners. This interface distinction is not a ranking. [Commands](../docs/QUICKSTART.md) · [Output paths and keys](../docs/OUTPUTS.md) · [Contributor contract](../docs/ADDING_METHOD.md).

`selected` is an internal GP strategy and historical reproduction option. Individual GP components are still needed by the ensemble. Original and early fixed-smoothing spline engines remain only for regression/reproduction. [Method evolution and retained history](../docs/HISTORY.md).
