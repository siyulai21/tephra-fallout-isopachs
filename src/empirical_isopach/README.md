# Shared empirical-isopach workflow

[Source map](../README.md) · [Regional protocol](../../docs/methods/regional-comparison.md) · [Adding methods](../../docs/ADDING_METHOD.md)

`tephra-isopachs compare` runs every registered regional model as a peer, followed by Bayesian nested isopachs, the spherical mixture and the [directional continuation from supported isopachs](models/directional/). It needs no per-eruption pseudo-points, smoothing inputs or chosen calculation radius.

| Path | Responsibility |
|---|---|
| [core/](core/README.md) | Accepted observations, shared likelihood/probability, survey geometry and scoring |
| [models/](models/README.md) | Separate model-family implementations and the active registry |
| [validation.py](validation.py) | Common automatic buffer, nested holdouts, predictive scores and deletion sensitivity |
| [region.py](region.py) | Shared observation hull, volume integration and correlated draws |
| [reporting.py](reporting.py) | Provenance, candidate tables and reports for all fitted methods |
| [visualization/](visualization/README.md) | Shared offline cartography, complete map frames, posterior-field views and reader comparisons |
| [support.py](support.py), [support_reporting.py](support_reporting.py) | Local evidence footprints, rejected contours, partial volumes and support reports |
| [global_reporting.py](global_reporting.py) | Spherical runner for the kernel posterior, bearing continuation and case report |
| [models/directional/](models/directional/README.md) | Directional interpolation, whole-Earth posterior, mass budgets, sensitivities and reporting |
| [models/mass_mixture/](models/mass_mixture/) | Kernel likelihood, bearing-dependent continuation, integrated mass and geographic tests |
| [models/nested/](models/nested/README.md) | Geographic contour ranks, thinning profiles, joint posterior and complete-deposit budgets |
| [cli.py](cli.py) | `methods`, `compare`, `global` and historical `fit` dispatch |

`spline.py` and `selection.py` are compatibility imports. Edit the cubic-spline implementation under `models/cubic_spline/`. Method-specific diagnostics belong to its adapter; shared reporting should not contain a branch for each new family.
