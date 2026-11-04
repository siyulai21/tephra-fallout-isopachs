# Trend-regularized cubic spline

[Active methods](README.md) · [Regional evidence](../comparisons/regional/README.md)

**Public identifier: `spline`.** A tensor-product cubic basis fits log thickness. Integrated derivative penalties control changes in slope and curvature; an area-normalized penalty shrinks toward a thinning trend estimated from training observations.

Geographic training scores select the trend family, knot intervals, tension, smoothing strength, trend strength and observation noise. Search expansion, boundary optima and numerical quadrature checks are recorded. Users supply geological evidence, not pseudo-point locations or smoothing settings.

The trend is estimated from the data; it is not independent evidence. Quadrature points approximate one continuous penalty rather than adding observations. Exact thicknesses and documented upper/lower bounds enter the observation likelihood. No GP residual field is fitted.

Joint coefficient draws provide conditional surface and volume intervals. They omit trend, parameter and selection uncertainty. Strong trend shrinkage can make intervals far too narrow; the reports retain held-out coverage and geographic-deletion sensitivity.

[Equations and search protocol](../docs/methods/cubic-splines.md) · [Shared regional evaluation](../docs/methods/regional-comparison.md) · [Implementation](../src/empirical_isopach/models/cubic_spline/README.md)
