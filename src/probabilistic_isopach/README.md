# Gaussian-process workflow

[Source map](../README.md) · [Model equations](../../docs/methods/gaussian-process.md) · [Current reports](../../reports/README.md)

This is the active package behind `tephra-isopachs`. It fits a directional trend plus a Matérn spatial residual model to log thickness, with documented upper/lower bounds handled through censored likelihoods. It uses no cubic-spline basis functions or knots.

| Files | Responsibility |
|---|---|
| `cli.py`, `__main__.py` | Run a named dataset or CSV and record settings |
| `catalog.py`, `data.py` | Resolve datasets and validate observations |
| `model.py` | Trend, spatial covariance, conditioning and censored likelihood |
| `validation.py` | Selection and nested geographic holdouts |
| `ensemble.py`, `distributions.py` | GP stacking and exact mixture calculations |
| `global_volume.py` | Primary complete-Earth integral and tail diagnostics |
| `volume.py` | Historical finite-region integration and conditional sampling |
| `contours.py` | Isopach geometry and resolution checks |
| `figures.py`, `publication.py`, `report.py` | Maps, tables, Markdown and numerical records |
| `assets/` | Bundled coastline data; see attribution |
| `regularized_spline.py` | Compatibility import; implementation is in [research/](../research/README.md) |

`selected` and `stacked` are both GP strategies. Probability intervals condition on estimated parameters; no global-volume probability interval is computed. See the [output guide](../../docs/OUTPUTS.md).
