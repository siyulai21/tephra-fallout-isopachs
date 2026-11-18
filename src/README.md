# Source-code map

[Project home](../README.md) · [Method catalogue](../methods/README.md) · [Agent routes](../AGENTS.md)

The active workflow is [empirical_isopach/](empirical_isopach/README.md). Its four families fit the same accepted evidence independently. Shared utilities provide consistent units, observation semantics, geometry, evaluation and reporting.

| Task | Edit here |
|---|---|
| Parse accepted data or change shared scoring | [core/](empirical_isopach/core/README.md) |
| Fit or select the regional GP ensemble | [models/gaussian_process/](empirical_isopach/models/gaussian_process/README.md) |
| Fit or tune the regional cubic spline | [models/cubic_spline/](empirical_isopach/models/cubic_spline/README.md) |
| Change directional interpolation, continuation or mass budgets | [models/directional/](empirical_isopach/models/directional/README.md) |
| Change spherical-mixture fans, priors or posterior | [models/mass_mixture/](empirical_isopach/models/mass_mixture/README.md) |
| Add a regional peer | [models/registry.py](empirical_isopach/models/registry.py) and the [method contract](../docs/ADDING_METHOD.md) |
| Change commands, support or report assembly | [Framework map](empirical_isopach/README.md) |

## Compatibility and reproduction

| Retained package | Why it is still here |
|---|---|
| [probabilistic_isopach/](probabilistic_isopach/README.md) | Thin imports of active implementations, historical GP-only commands, Earth integration helpers and map assets reused by current reports |
| [spline_isopach/](spline_isopach/README.md) | Original spline numerical engine and regression references; the unused desktop GUI has been removed |
| [research/](research/README.md) | First fixed-smoothing spline experiment, needed by historical benchmarks and regression checks |

These are not additional default families. Do not delete shared Earth/plotting helpers just because their package name is historical. `Laea.py` still supplies source-coordinate preparation for reproduction. [Tests](../tests/README.md) · [Benchmark commands](../benchmarks/README.md) · [History](../docs/HISTORY.md).
