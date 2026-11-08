# Historical GP workflow and compatibility imports

[Source map](../README.md) · [Active GP implementation](../empirical_isopach/models/gaussian_process/README.md) · [Historical output guide](https://github.com/siyulai21/tephra-fallout-isopachs/blob/41b8f2f28ac8b1c877810b782005ff243e101c63/archive/reference/historical-gp-outputs.md)

The active GP ensemble has moved into the shared framework's `models/gaussian_process/` directory. Shared data/probability/scoring utilities live in `empirical_isopach/core/`. Old imports remain valid through thin wrappers; there is one implementation of each solver.

| Retained files | Role |
|---|---|
| `model.py`, `ensemble.py`, `validation.py` | Compatibility exports of active GP implementations and shared scoring |
| `catalog.py`, `data.py`, `distributions.py` | Compatibility exports of shared core utilities |
| `cli.py`, `__main__.py` | Historical GP-only `fit`; `compare` dispatches to the active workflow |
| `global_volume.py`, `volume.py` | Historical complete-Earth and rectangular-domain calculations |
| `contours.py`, `publication.py`, `report.py` | Historical GP report and contour exports |
| `figures.py`, `assets/` | Historical plotting and bundled coastline/style resources reused by current reports |
| `regularized_spline.py` | Compatibility export of the first [research extension](../research/README.md) |

This directory is not a second active method catalogue. Only the [registry](../empirical_isopach/models/registry.py) determines the families in a new regional comparison.
