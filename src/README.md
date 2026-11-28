# Source-code map

[Project home](../README.md) · [Method catalogue](../docs/methods/README.md) · [Agent routes](../AGENTS.md)

The active workflow is [empirical_isopach/](empirical_isopach/README.md). Its model families fit the same accepted evidence independently. Shared utilities provide consistent units, observation semantics, geometry, evaluation and reporting.

| Task | Edit here |
|---|---|
| Parse accepted data or change shared scoring | [core/](empirical_isopach/core/README.md) |
| Fit or select the regional GP ensemble | [models/gaussian_process/](empirical_isopach/models/gaussian_process/README.md) |
| Fit or tune the regional cubic spline | [models/cubic_spline/](empirical_isopach/models/cubic_spline/README.md) |
| Change directional interpolation, continuation or mass budgets | [models/directional/](empirical_isopach/models/directional/README.md) |
| Change nested areas, contour shapes or distal laws | [models/nested/](empirical_isopach/models/nested/README.md) |
| Change spherical-mixture fans, priors or posterior | [models/mass_mixture/](empirical_isopach/models/mass_mixture/README.md) |
| Add a regional peer | [models/registry.py](empirical_isopach/models/registry.py) and the [method contract](../docs/ADDING_METHOD.md) |
| Change commands, support or report assembly | [Framework map](empirical_isopach/README.md) |

## Shared dependencies

The WGS84 Earth-domain implementation is part of the recorded identity of the global models. The projection helper sets the vent-centred coordinates for all fits. The older GP import paths also expose the active component model and probability utilities. [Earth geometry and compatibility imports](probabilistic_isopach/README.md) · [Coordinate transform](Laea.py)

The original spline, its first trend-penalty experiment and the GP-only command can be reproduced from the release listed in the [project history](../CHANGELOG.md#earlier-results).

Wrapped angular covariance fits cumulative regional thinning and a conditional source-area excess. Its runner writes correlated Earth-wide draws, map integrals and geographic deletions. [Implementation map](empirical_isopach/models/wrapped_angular/README.md)
