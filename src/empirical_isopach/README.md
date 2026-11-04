# Shared empirical-isopach workflow

[Source map](../README.md) · [Regional protocol](../../docs/methods/regional-comparison.md) · [Adding methods](../../docs/ADDING_METHOD.md)

`tephra-isopachs compare` runs every registered model as a peer. It has no default winner and needs no per-eruption pseudo-points or smoothing inputs.

| Path | Responsibility |
|---|---|
| [core/](core/README.md) | Accepted observations, shared likelihood/probability, survey geometry and scoring |
| [models/](models/README.md) | Separate model-family implementations and the active registry |
| [validation.py](validation.py) | Common automatic buffer, nested holdouts, predictive scores and deletion sensitivity |
| [region.py](region.py) | Shared observation hull, volume integration and correlated draws |
| [reporting.py](reporting.py) | Provenance, candidate tables and reports for all fitted methods |
| [cli.py](cli.py) | `methods`, `compare` and historical `fit` dispatch |

`spline.py` and `selection.py` are compatibility imports. Edit the cubic-spline implementation under `models/cubic_spline/`. Method-specific diagnostics belong to its adapter; shared reporting should not contain a branch for each new family.
