# Source-code map

[Project home](../README.md) · [Method guide](../docs/METHODS.md)

`tephra-isopachs compare` runs two peer methods with common regional validation. Existing imports and legacy commands remain available for historical reproduction.

| Directory | Role |
|---|---|
| [empirical_isopach/](empirical_isopach/README.md) | Active automatic spline, common regional evaluation, integration and comparison reports |
| [probabilistic_isopach/](probabilistic_isopach/README.md) | Active GP-ensemble fitter and shared probability/data utilities; also the legacy GP/global CLI |
| [spline_isopach/](spline_isopach/README.md) | Original cubic spline and optional legacy GUI; historical reproduction |
| [research/](research/README.md) | First fixed-smoothing trend-penalty experiment; historical research |

The active spline uses its own cubic basis, trend and penalized mixed-likelihood solver. It never fits a GP residual. The GP ensemble still needs individual GP component fitters internally. Shared parsing, probability scoring and region tools do not make these the same statistical method.

`Laea.py` supplies projection helpers. New method families should use the same accepted observations, geographic splits, prediction contract and regional volume definition. [Benchmark commands](../benchmarks/README.md) · [Tests](../tests/README.md).
