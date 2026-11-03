# Which method does what?

[Project home](../README.md) · [Documentation](README.md) · [Source-code map](../src/README.md)

**Every current eruption report uses a Gaussian-process model.** The repository also preserves two cubic-spline implementations as research alternatives. They are not three stages of one calculation.

| Implementation | How the surface is represented | Controls | Current eruption reports? |
|---|---|---|---|
| [Gaussian process](methods/gaussian-process.md) | Directional thinning trend plus a Matérn spatial residual field | Geographic selection of thinning family, correlation length and noise ratio; smoothness class is fixed | **Yes** |
| [Original cubic spline](methods/cubic-splines.md) | Cubic B-spline basis coefficients | Tension, roughness and knot layout | No |
| [Trend-regularized cubic spline](methods/cubic-splines.md) | Original spline plus a penalty toward a fitted thinning trend | Original controls plus a normalized, data-selected trend penalty | No |

## Labels used in the reports

- **`selected`**: one GP candidate selected using geographic predictive scores. It does not mean the original spline.
- **`stacked`**: an experimental weighted ensemble of GP candidates. It is not a combination of GP and spline models.
- **Exponential / power-law**: the broad thinning trend within the GP model, not a different interpolation method.
- **Global / regional**: the volume integration domain, not the model family.

Both current strategies use the same accepted observations and explicit treatment of bounds. Their details are in the [GP model](methods/gaussian-process.md) and [ensemble method](methods/gp-ensemble.md). The primary [volume calculation](methods/global-volume.md) integrates the fitted field over the complete Earth, while regional calculations remain diagnostics.

## What remains an assumption?

The GP smoothness class, finite candidate grid, directional trend and treatment of preservation remain modeling choices. Selection is reproducible but does not eliminate extrapolation assumptions. Reported uncertainty is conditional on fitted parameters; no global-volume probability interval is estimated.

The spline extension addresses arbitrary pseudo-point influence by normalizing a trend penalty. It does not turn predictions into independent measurements, jointly optimize every spline control, or provide full uncertainty estimates.

## Where is the comparison evidence?

The [validation catalogue](validation/README.md) separates dated development benchmarks from current eruption reports. Earlier four-eruption experiments include a fixed-default spline comparison; current selected-versus-stacked reports compare GP strategies. Neither establishes that the GP approach outperforms a fully tuned spline for every eruption. [Report catalogue](../reports/README.md).
