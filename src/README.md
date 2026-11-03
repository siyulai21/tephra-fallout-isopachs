# Source-code map

[Project home](../README.md) · [Method comparison](../docs/METHODS.md)

**`tephra-isopachs` runs the Gaussian-process workflow. Every published eruption report uses that family.** Spline code is retained for research, not invoked as a step in that workflow.

| Directory | Role | Used for current reports? |
|---|---|---|
| [probabilistic_isopach/](probabilistic_isopach/README.md) | GP fitting, validation, integration and reporting | Yes |
| [spline_isopach/](spline_isopach/README.md) | Original cubic B-spline model and optional legacy GUI | No |
| [research/](research/README.md) | Experimental trend-regularized cubic spline | No |

`Laea.py` supplies shared projection helpers. Existing `src.*` package paths are retained so commands and downstream imports keep working.

## Dependencies between methods

- The GP workflow uses its own spatial model; it does not import or call the spline solver.
- The original spline fits cubic basis coefficients with tension and roughness penalties.
- The research extension uses the original spline solver and borrows trend/noise estimates from the GP module. Its added target is the fitted trend, not the GP residual surface.
- `probabilistic_isopach/regularized_spline.py` is only a compatibility import. New research code should import from `src.research.trend_regularized_spline`.

See [benchmarks](../benchmarks/README.md) and [tests](../tests/README.md). Folder placement does not imply one method has been proven superior.
