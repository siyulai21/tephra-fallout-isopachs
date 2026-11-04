# Active empirical models

[Project home](../README.md) · [Compare results](../comparisons/README.md) · [Add a model](../docs/ADDING_METHOD.md)

**Two active methods; equal standing.** Each fits its own surface and calculates its own volume from that surface. Shared evidence and evaluation make their results comparable.

| Method | How the surface is fitted | Automatic choices | Guide |
|---|---|---|---|
| GP ensemble (`stacked`) | Directional thinning trends plus correlated Gaussian-process residual fields, combined by predictive stacking | Component candidates and weights from geographic training folds | [GP ensemble](gaussian-process.md) |
| Trend-regularized cubic spline (`spline`) | Tensor-product cubic basis with integrated derivative and fitted-trend penalties | Trend, knots, tension, smoothing, trend strength and residual noise | [Cubic spline](cubic-spline.md) |

Neither needs invented pseudo-observations. Numerical integration points evaluate integrals and penalties; they do not supply additional evidence. Automation removes per-eruption hand tuning, not the model's fixed assumptions or search limits.

The [code registry](../src/empirical_isopach/models/registry.py) is the executable list of active families. `tephra-isopachs methods` displays it. Adding a registry entry includes the method in fitting, geographic evaluation, maps and comparison tables; [the contributor contract](../docs/ADDING_METHOD.md) specifies the statistical obligations.

`selected` remains an internal GP strategy and historical reproduction option, not another public family. The original spline and first trend-penalty experiment are historical implementations. [Method history](../docs/METHODS.md) · [Source map](../src/README.md).
