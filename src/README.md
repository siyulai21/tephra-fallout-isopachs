# Source-code map

[Project home](../README.md) · [Method catalogue](../methods/README.md) · [Add a family](../docs/ADDING_METHOD.md)

The active workflow lives in **[empirical_isopach/](empirical_isopach/README.md)**. Its two model families have separate implementations; they share observation semantics, validation, regional integration and reporting.

| Directory | Role |
|---|---|
| [empirical_isopach/core/](empirical_isopach/core/README.md) | Common data, probability, geography and scoring utilities |
| [empirical_isopach/models/](empirical_isopach/models/README.md) | Active registry, GP ensemble and automatic cubic spline |
| [empirical_isopach/](empirical_isopach/README.md) | Shared comparison runner, volume integration, maps and reports |
| [probabilistic_isopach/](probabilistic_isopach/README.md) | Old import wrappers, historical GP/global workflow and reused map assets |
| [spline_isopach/](spline_isopach/README.md) | Historical original spline and optional desktop GUI |
| [research/](research/README.md) | Historical first fixed-smoothing trend-penalty experiment |

The GP does not call the spline solver. The spline does not fit a GP residual. GP components are internal ingredients of one public ensemble, not extra public methods. Historical source packages are retained for reproduction and are not active registry entries.

`Laea.py` supplies projection helpers. [Tests](../tests/README.md) · [Benchmark commands](../benchmarks/README.md).
