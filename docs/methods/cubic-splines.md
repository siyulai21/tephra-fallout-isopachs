# Cubic-spline methods

[Method overview](../METHODS.md) · [Source map](../../src/README.md) · [Research benchmark](../validation/initial-benchmark.md)

These are retained research alternatives. Neither produced the current eruption reports; those use the Gaussian-process workflow.

## Original cubic spline

The [original implementation](../../src/spline_isopach/README.md) represents log thickness as a cubic B-spline surface. It balances measurement misfit against tension and roughness penalties. It is already a regularized estimator: “regularized spline” does not mean the original lacked smoothing.

Tension mixes first- and second-derivative penalties. The roughness setting scales their strength. Knot counts and spacing follow deterministic rules based on the domain, sample count and site spacing; a plotting grid is a separate numerical choice. The existing implementation includes corrected banded-system assembly and solver checks.

## Why add a trend penalty?

Manual pseudo-points can make results depend on their positions, assigned thicknesses and number. The experimental extension retains the cubic basis and original penalties, but adds a normalized penalty toward a thinning trend estimated from real observations. Increasing the number of integration sites approximates the same penalty rather than automatically increasing its total strength.

The implementation is [src/research/trend_regularized_spline.py](../../src/research/trend_regularized_spline.py). It borrows trend/noise estimates from the GP module; it does not use the GP residual surface as its target. The GP report workflow does not call this extension.

## Mathematical definition

For z(s)=B(s)c, preserve the data loss and original tension/roughness penalty and add an area-averaged penalty toward the fitted trend:

\[
J(c)=\frac12\sum_i\frac{[\log Y_i-B_ic]^2}{\sigma^2+v_i}
+\frac12c^TP_{\tau,\rho}c
+\frac\lambda{2A^2}\int_D[B(s)c-m(s)]^2\frac{ds}{|D|}.
\]

Midpoint quadrature replaces the integral by a weighted sum with q_j>=0 and sum(q_j)=1. Virtual-point precision is **lambda*q_j/A²**. Equivalently its artificial-point variance is A²/(lambda*q_j). Refining the grid approximates the same integral instead of increasing prior information. Duplicating each location with half its weight leaves the normal equations exactly unchanged.

The implementation adds B_Qᵀ diag(lambda*q/A²) B_Q and the corresponding target vector to the complete banded spline system. Domain, knots, normalization and data-error estimates derive from real data, never from virtual points. Tests verify exact replication invariance and approximate quadrature convergence.

`select_spline_prior` selects lambda from {0,1,10,100} with spatially held-out squared log errors; zero is the no-prior control. The trend is exponential and tension=0.25, roughness=3.2 are fixed to isolate the prior question. A boundary optimum is a reason to examine an expanded sensitivity range, not proof of a global optimum.

Since m and A are learned from the training observations, this is **data-adaptive regularization**, not independent prior evidence. All thickness-dependent fitting is repeated within each fold. A fixed known survey rectangle uses all sampling coordinates; this is declared design information. The selection diagnostics are not nested estimates of performance. This estimator provides point predictions only and refuses outside-domain predictions.


## What has and has not been established

The [initial benchmark](../validation/initial-benchmark.md) compares an original-default spline and the GP on four development datasets, and separately reports spline-prior selection diagnostics. The baselines are not all independently tuned, and the prior-selection scores are not outer validation. They do not establish universal superiority or a spline-versus-GP result for the current Minoan/YTT reports.

The extension supplies point estimates only, rejects censored inputs and refuses extrapolation outside its fitted rectangle. Original tension and roughness remain fixed during prior-strength selection. This is an experiment in reproducible constraints, not a full posterior treatment of uncertainty.
