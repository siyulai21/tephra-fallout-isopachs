# Active empirical models

[Project home](../README.md) · [Compare results](../comparisons/README.md) · [Add a model](../docs/ADDING_METHOD.md)

**Two regional peers and a global companion.** GP and spline each fit their own regional surface. The mass-normalized model estimates whole-deposit fallout under explicit generic assumptions. Shared evidence and geographic evaluation make predictive performance comparable; a regional volume is not a global volume.

| Method | How the surface is fitted | Automatic choices | Guide |
|---|---|---|---|
| GP ensemble (`stacked`) | Directional thinning trends plus correlated Gaussian-process residual fields, combined by predictive stacking | Component candidates and weights from geographic training folds | [GP ensemble](gaussian-process.md) |
| Trend-regularized cubic spline (`spline`) | Tensor-product cubic basis with integrated derivative and fitted-trend penalties | Trend, knots, tension, smoothing, trend strength and residual noise | [Cubic spline](cubic-spline.md) |
| Global spherical mixture (`global`) | Positive jointly fitted fans with whole-Earth mass normalization and numerical refinement checks | Full posterior for volume, core thinning, tails, fan geometry and noise; fixed prior sensitivities | [Global fallout](global-fallout.md) |

None requires invented pseudo-observations. Numerical integration points evaluate integrals and penalties; they do not supply additional evidence. Automation removes per-eruption hand tuning, not the model's fixed assumptions or search limits.

The regional peers pass through the same [support and complete-contour audit](../docs/methods/supported-isopachs.md). A smooth GP prediction and a closed spline contour both require spatial evidence. The audit reports local partial volumes, refuses unsupported closure, and keeps proximal/distal continuations as explicit scenarios. [Results across all eruptions](../comparisons/support/README.md).

The [code registry](../src/empirical_isopach/models/registry.py) lists the regional families. The global companion has a separate joint-posterior interface in [mass_mixture](../src/empirical_isopach/models/mass_mixture/). `tephra-isopachs methods` describes both scopes; `compare` runs both. The [contributor contract](../docs/ADDING_METHOD.md) specifies the statistical obligations.

`selected` remains an internal GP strategy and historical reproduction option, not another public family. The original spline and first trend-penalty experiment are historical implementations. [Method history](../docs/METHODS.md) · [Source map](../src/README.md).
