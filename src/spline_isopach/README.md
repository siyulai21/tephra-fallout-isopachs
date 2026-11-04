# Original cubic-spline implementation

[Source map](../README.md) · [Spline methods](../../docs/methods/cubic-splines.md) · [Experimental extension](../research/README.md)

This package retains the cubic B-spline approach: fit log thickness with basis functions and tension/roughness penalties. It includes numerical corrections recorded in the [change history](../../CHANGELOG.md); “original” identifies the method family, not an untouched copy of the earliest code.

**It does not generate the current eruption reports.** Those now use the [GP-ensemble / automatic-spline comparison](../empirical_isopach/README.md); this is the historical solver.

| File | Role |
|---|---|
| `lsf_bsplines2d.py` | Basis evaluation, complete banded system, penalties and surface evaluation |
| `Model.py` | Compatibility functions for the spline API |
| `__main__.py` | Optional legacy desktop GUI; requires the extra mapping dependency |

The spline's domain and knot layout are separate from a display grid. The [initial research benchmark](../../docs/validation/initial-benchmark.md) contains fixed-default comparisons and their limitations. It does not establish that GP outperforms a fully tuned spline on every eruption.
