# Shared empirical-isopach workflow

[Source map](../README.md) · [Regional protocol](../../docs/methods/regional-comparison.md) · [Adding methods](../../docs/ADDING_METHOD.md)

`tephra-isopachs compare` runs every registered regional model as a peer, followed by the probabilistic global companion. It needs no per-eruption pseudo-points, smoothing inputs or chosen calculation radius.

| Path | Responsibility |
|---|---|
| [core/](core/README.md) | Accepted observations, shared likelihood/probability, survey geometry and scoring |
| [models/](models/README.md) | Separate model-family implementations and the active registry |
| [validation.py](validation.py) | Common automatic buffer, nested holdouts, predictive scores and deletion sensitivity |
| [region.py](region.py) | Shared observation hull, volume integration and correlated draws |
| [reporting.py](reporting.py) | Provenance, candidate tables and reports for all fitted methods |
| [support.py](support.py), [support_reporting.py](support_reporting.py) | Local evidence footprints, rejected contours, partial volumes and support reports |
| [global_reporting.py](global_reporting.py) | Whole-Earth posterior, sensitivity, sampling diagnostics and global reports |
| [models/mass_mixture/](models/mass_mixture/) | Mass-normalized fans, priors, posterior inference and geographic validation |
| [cli.py](cli.py) | `methods`, `compare`, `global` and historical `fit` dispatch |

`spline.py` and `selection.py` are compatibility imports. Edit the cubic-spline implementation under `models/cubic_spline/`. Method-specific diagnostics belong to its adapter; shared reporting should not contain a branch for each new family.
