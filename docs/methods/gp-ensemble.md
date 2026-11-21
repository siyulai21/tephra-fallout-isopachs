# Regional Gaussian-process ensemble

The ensemble averages plausible thinning trends and spatial correlation structures. Their weights come from predictions at geographically withheld training sites. Field measurements and defensible bounds enter the likelihood. Unsampled locations are prediction targets. [Observation protocol](sites-without-measured-thickness.md) · [Component equations](gaussian-process.md)

## Estimate weights using withheld regions

Each component retains the existing constrained directional mean and Matérn spatial residual model. The candidate grid is fixed: two tail families, three correlation-length factors and three residual-noise ratios. All scale estimation and fitting happen again inside each training fold. Colocated observations stay in one geographic fold. A shared geographic buffer excludes nearby training locations.

For observation \(i\), candidate \(k\) supplies an out-of-fold predictive density \(p_{ik}\). A documented upper limit contributes its predictive CDF instead; a documented lower bound contributes the survival probability. Retain the three lowest equal-region negative-log-score candidates per tail family, at most six total. Fit weights by

\[
\widehat w=\arg\max_{w_k\geq0,\ \sum_k w_k=1}
\sum_{b=1}^{B}\frac1B\sum_{i\in b}\frac1{n_b}
\log\!\left(\sum_k w_k p_{ik}\right).
\]

Equal region mass prevents a densely sampled geographic fold from automatically dominating this objective. It does not solve preferential sampling within a region. The simplex optimization is deterministic; weights at or below `1e-7` are dropped and the retained weights renormalized. Removed numerical mass is reported. Ties follow the fixed candidate order.

Refit retained components on all accepted observations. The result is a predictive mixture, **not posterior probabilities that the models are true**. A family may receive zero weight; the algorithm does not force both families to survive. Its benefit can also come from averaging correlation-length or noise candidates within one family.

Whole-fold Dirichlet reweighting (64 draws, fixed seed 1729) refits weights to the saved out-of-fold scores. Reported 5–95% ranges reveal weight instability. They condition on the fitted candidate predictions and screening decision; they are neither Bayesian credible intervals nor frequentist confidence intervals. They are not propagated as an additional source of surface uncertainty.

The statistical basis is predictive-distribution stacking under proper scoring rules, as developed by [Yao, Vehtari, Simpson and Gelman (2018)](https://doi.org/10.1214/17-BA1091). This implementation uses actual geographic refits and conditional Gaussian/Laplace components, rather than their leave-one-out posterior approximation. Geographic validation follows the dependence concerns discussed by [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881).

## Map thickness and integrate the observation hull

Let component log thickness have mean \(\mu_k(x)\) and covariance \(C_k\). The predictive field is a mixture of these joint distributions. Its expected thickness is

\[
E[T(x)]=\sum_k w_k\exp\{\mu_k(x)+C_k(x,x)/2\}.
\]

Pointwise medians and interval endpoints solve the mixture CDF, rather than using a Gaussian approximation to its moments. Consequently the median isopach surface can differ from the exponential of the mean log surface. Covariance exports use the law of total covariance, including between-component mean differences.

Each regional volume draw uses one component for the whole observation hull. Draw a correlated log-thickness field from that component, exponentiate it and integrate over the hull. The quantiles of those complete-volume draws give the conditional interval. The integral of expected thickness and the integral of pointwise median thickness are also reported.

These calculations represent conditional spatial uncertainty and retained-model disagreement. They do **not** integrate fitted mean/covariance parameters, learned weights, candidate screening, vent error, field preservation or sampling-process uncertainty. Power-tail extrapolation and finite-region volume remain assumption-dependent. Numerical integration refinement does not establish geological correctness or convergence of sampled volume quantiles.

## Observation discrepancy

An empirical diagnostic estimates nonnegative extra variance \(\tau^2\) by scoring the same training out-of-fold mixture after replacing each observation variance \(v_{ik}\) with \(v_{ik}+\tau^2\). This is additional independent variation for a **new observation**, not latent spatial structure.

The numerical search uses 81 logarithmically spaced positive variance values plus zero, then bounded refinement around the best grid value. The upper bound is `max(1, 4 × mean squared out-of-fold detection error)`. Both endpoints are considered and boundary hits reported. This deterministic one-dimensional search is an approximation, not a proof of a global optimum for every mixture. The cap, grid, candidate screen and covariance families are declared algorithmic choices; they are not elicited separately for each eruption.

The diagnostic estimates extra variation in a new measurement. Latent thickness, the primary observation intervals and volume use the fitted mixture. The training record includes the estimated variance and the score before and after adjustment.

## Run and validate

`tephra-isopachs compare Minoan --regional-only --output results/new-minoan-regional` fits the GP ensemble and the cubic spline on the same hull. Outer geographic holdouts test the full fitting procedure, including selection of weights inside each training set. The published comparison reports the scores, interval coverage and numerical warnings for each registered eruption. [Regional comparison and known-deposit tests](../../comparisons/regional/README.md)
