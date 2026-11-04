# Automatic trend-regularized cubic spline

[Method overview](../METHODS.md) · [Regional evidence](../../reports/regional-methods/README.md) · [Implementation](../../src/empirical_isopach/spline.py)

The active spline is a cubic tensor-product B-spline model of **log thickness**, fitted to real measurements and documented bounds. It is a peer of the GP ensemble. It does not call a GP surface fitter or require manually placed pseudo-points, knot spacing, tension, roughness or trend strength.

“Automatic” means one documented, versioned algorithm across eruptions. It does not mean assumption-free or guaranteed correct in unobserved areas.

## Model and likelihood

Let f(x)=B(x)c. Exact observations satisfy log(Y_i) ~ Normal(f(x_i), sigma²+v_i), with optional known log-scale error variance v_i. An upper limit contributes Phi((log L_i-f_i)/s_i); a lower bound contributes Phi((f_i-log L_i)/s_i). No bound becomes an exact thickness and no unsampled location becomes zero.

The objective is

\[
J(c)=-\log p(\mathrm{observations}\mid c)
+\frac{\alpha}{2A^2}\{\tau\langle\|\nabla_u f\|^2\rangle_D
+(1-\tau)\langle f_{u_1u_1}^2+2f_{u_1u_2}^2+f_{u_2u_2}^2\rangle_D\}
+\frac{\lambda}{2A^2}\langle[f-m]^2\rangle_D.
\]

Brackets denote **area averages**. Coordinates u=(x-x_min)/L use one common length L, the rectangle's longest side. This makes penalties invariant to coordinate units. A is training RMS residual around the fitted trend, floored at 0.1 log units. Alpha controls smoothing; tau mixes gradient and curvature penalties; lambda controls attraction to the trend. Alpha is a new explicit multiplier, **not** the legacy `rou` parameter.

An exponential or power directional trend is fitted to training exact measurements by constrained least squares, requiring outward decay along every ray. The power exponent is not forced above two because no infinite-plane volume is requested. Bounds enter the final spline likelihood, never the trend as substituted exact values. A flexible spline can depart from this single directional target and represent multiple fallout fans.

The target and scale use the same measurements as the spline: this is **data-adaptive regularization**, not independent prior evidence. Every response-dependent step is repeated within training folds.

## Basis and numerical solution

Clamped cubic knots are equally spaced on each axis. The selected number of intervals refers to the rectangle's longer side; the other axis follows its aspect ratio. The implementation uses [SciPy's cubic B-spline basis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html).

Mass, gradient and full thin-plate curvature matrices use tensor products of one-dimensional Gauss integrals. Four nodes per knot span exactly integrate their polynomial products; six are used by default. The nonpolynomial trend target is checked against twelve nodes in each final fit. Area weights sum to one. Duplicating numerical points with split weights cannot increase information. Prediction-grid density does not control fitting.

Exact-data fits solve the complete normal equations with Cholesky factorization. Mixed exact/bounded fits use damped Newton steps, an Armijo line search and a final Newton-decrement check. Invalid domains, insufficient or collinear exact measurements, non-convergence and outside-rectangle predictions fail explicitly. No guessed thickness clipping or silent zero extension is used.

## Automatic selection

Every training set evaluates the same grid exhaustively:

| Choice | Initial candidates |
|---|---|
| Trend | exponential, power |
| Knot intervals on longer axis | 2, 4, 6, 8 |
| Tension tau | 0, 0.5, 1 |
| Smoothing alpha | 10^-6, 10^-4, 0.01, 1, 100 |
| Trend strength lambda | 0, 0.01, 1, 100, 10,000 |
| Noise / trend-residual RMS | 0.25, 0.75, 1.5, 3 |

Noise has a floor of 0.05 log units; known measurement error is added in quadrature. The 2,400 initial candidates jointly vary every setting. A boundary winner in each trend family triggers one exhaustive expanded grid: alpha can extend to 10^-8 or 10,000; lambda to 1,000,000; noise ratio to 0.125 or 6; intervals to 1 or 12. Zero trend strength and pure gradient/curvature penalties are legitimate endpoints.

Selection minimizes **equal-geographic-fold negative log predictive score**, including bound probabilities. A candidate must fit every fold. Fixed order resolves ties. Reports retain every candidate, failure, fold score and remaining boundary hit. The minimum is over evaluated grids, not a guaranteed continuous optimum. Search limits and model families are fixed algorithmic assumptions, not inputs elicited for each eruption.

Penalized B-splines are established methods; see [Eilers and Marx (1996)](https://doi.org/10.1214/ss/1038425655). This implementation uses integrated derivative penalties rather than their coefficient-difference penalty.

## Validation and uncertainty

The [regional protocol](regional-comparison.md) uses four outer and three inner geographic folds. Every fitting and selection step excludes outer responses. The final fit uses four tuning folds. Full survey coordinates define the common rectangle and hull as known design information; thicknesses do not define the region.

The inverse coefficient Hessian gives a conditional Gaussian coefficient distribution for exact data and a Laplace approximation for bounds. With fixed estimated trend, noise, knots and penalties, the exact-data covariance is the Gaussian penalized-model covariance. It is **not** an unconditional frequentist confidence matrix.

One coefficient draw determines a complete correlated surface. Exponentiating and integrating those draws produces conditional regional volume intervals. Observation noise enters predictions for a new measured thickness, but not latent fields or volumes. Expected volume and the integral of pointwise medians are reported separately.

Intervals omit fitted-trend, noise, hyperparameter/selection, preservation, sampling-process and vent-location uncertainty. Strong trend shrinkage can produce extremely narrow conditional intervals without establishing precise volume. Reports flag this case, display fully retuned geographic-deletion ranges separately and assess empirical coverage. Deletion ranges are sensitivity diagnostics, not confidence intervals. Unfavorable validation outcomes remain visible.

## Historical implementations

The [original spline](../../src/spline_isopach/README.md) and [first trend-penalty experiment](../../src/research/README.md) remain for historical reproduction. The original already included tension/roughness regularization. The first extension kept smoothing fixed, rejected bounds, returned point predictions only and borrowed a GP module's trend estimator.

Version 0.8 retains the cubic-basis and normalized-penalty ideas but implements the basis, derivatives and mixed likelihood afresh in `src/empirical_isopach/`. It does not claim numerical equivalence to the historical solver or invention of penalized splines. Old imports and report artifacts remain available.
