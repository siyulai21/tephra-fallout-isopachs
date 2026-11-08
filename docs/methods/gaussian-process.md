# Gaussian-process thickness model

[Method overview](../METHODS.md) · [Documentation](../README.md) · [Implementation](../../src/empirical_isopach/models/gaussian_process/README.md)

**Used by the GP side of the current regional comparison and historical GP-only reports.** `selected` chooses one GP configuration; `stacked` combines GP configurations. Neither uses cubic splines.

## Identifiability comes first

The objective is to eliminate hand placement and guessed thickness of pseudo-points. It is impossible to eliminate every extrapolation assumption: a nonnegative smooth bump supported away from all measured sites can be added to any compatible surface, with arbitrary amplitude and volume, without changing a measurement. Smoothness without quantitative bounds does not solve this. Cross-validation cannot validate a wholly unobserved sector.

The implementation removes manual pseudo-data dependence, not modelling assumptions. It does not automatically recover hidden lobes or resolve preservation history.

## Prior art and proposed contribution

[Engwell et al. (2015)](https://doi.org/10.1007/s00445-015-0942-y) provide the spline basis of this repository. [Rougier et al. (2022)](https://doi.org/10.1093/gji/ggac131) explicitly encode expert judgement using distal pseudo-measurements. [Yang and Bursik (2016)](https://doi.org/10.1007/s00445-016-1061-0) already combine a thinning trend with kriging and identify manually chosen wind direction as a limitation.

Gaussian-process conditioning follows [Rasmussen and Williams](https://gaussianprocess.org/gpml/chapters/RW2.pdf). Virtual derivative constraints also have established precedent in [Riihimäki and Vehtari (2010)](https://proceedings.mlr.press/v9/riihimaki10a.html). Spatial validation is motivated by [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881).

The GP workflow combines typed evidence, fitted direction, exclusion of invented observations from validation, correlated external constraints and extrapolation diagnostics. The separate trend-regularized cubic-spline experiment is documented in [cubic-spline methods](cubic-splines.md). This is a research implementation, not a claim of a new statistical theory or established publication novelty.

## Trend and spatial covariance

Let s=(x,y) be vent-centred WGS84 Lambert azimuthal equal-area coordinates in kilometres, and z=ln T for thickness T in centimetres:

\[
z(s)=m_\theta(s)+g(s),\quad g\sim GP(0,k),\quad
\log Y_i\mid z_i\sim N(z_i,\sigma^2+v_i).
\]

Known additional log-error variance is v_i; sigma is residual local variation plus unmodelled measurement error, not identifiable as pure instrument error. The log model guarantees positive thickness but does not model structural zero deposition.

With r=||s||, u=s/r (zero at the vent), and training median positive radius r0,

\[
m(s)=a-bh(r)+(c_xu_x+c_yu_y)h(r).
\]

Candidates use h=r/r0 (exponential thinning) or h=ln(1+r/r0) (power tail). Constrained least squares on training detections estimates a,b,c with b>=||c||+10^-6 for exponential thinning and b>=||c||+2+10^-6 for a finite planar power-tail integral. No wind direction is supplied manually. This directional term is phenomenological, not an atmospheric reconstruction. Only the mean trend decreases on rays; spatial residuals can produce secondary maxima.

The Matérn 3/2 kernel is

\[
k(s,t)=A^2(1+\sqrt3d/\ell)e^{-\sqrt3d/\ell}.
\]

Each training fit sets A=max(log-residual RMS,0.1). Candidate ell values are {0.25,0.75,1.5} times the median positive pairwise detection distance. Candidate sigma values are max({0.25,0.75,1.5} A,0.05). These finite grids and floors are declared defaults, not physical constants or a globally optimal search. Correlation is isotropic in the projected plane; long-range projection distortion remains a limitation.

For uncensored data, C=K+diag(sigma²+v). Prediction at any query set Q is

\[
\mu_Q=m_Q+K_{QD}C^{-1}(y-m_D),\qquad
\Sigma_Q=K_{QQ}-K_{QD}C^{-1}K_{DQ}.
\]

**Sigma_Q matters.** Predicted values are correlated and cannot be reinserted as independent observations. `virtual_predictions.npz` exports the joint distribution. Evaluating additional query sites adds no evidence and does not change earlier predictions.

This is a two-stage empirical-Bayes-style procedure. The mean and covariance parameters are estimated and then held fixed. Intervals omit uncertainty in those estimates and model selection; they are conditional pointwise intervals, not full Bayesian credible intervals or simultaneous contour bands.

## Censored observations and independent soft constraints

A documented upper limit L_i has likelihood

\[
P(Y_i\le L_i\mid z_i)=\Phi\left((\log L_i-z_i)/\sqrt{\sigma^2+v_i}\right).
\]

It is neither an exact zero nor an invented midpoint L_i/2. The mixed Gaussian/probit likelihood uses a whitened Laplace approximation, stable log-CDF evaluation, and a checked convergence gradient. Derivative tests compare analytic gradients/Hessians with numerical derivatives. A loose bound is nearly uninformative; a low bound lowers local predictions.

Use this only when actual field detection and preservation information justifies an upper limit on the target thickness. An eroded or unvisited site does not establish a bound on original deposition. Trace observations without documented bounds are not converted to arbitrary positive numbers. Old CSVs with already-substituted positive trace values require provenance review.

For genuinely independent external evidence e=Hz+eta, eta~N(0,R), Gaussian conditioning gives

\[
\mu'=\mu+\Sigma H^T(H\Sigma H^T+R)^{-1}(e-H\mu),
\]
\[
\Sigma'=\Sigma-\Sigma H^T(H\Sigma H^T+R)^{-1}H\Sigma.
\]

`condition_independent_evidence` implements point-selection H with a full covariance R. A shared-source model R=diag(v)+tau²11ᵀ prevents replication from eliminating shared uncertainty. Exactly repeated evidence with fully correlated errors yields the same result as one copy. A pseudoinverse handles such duplicates and inconsistent singular evidence is rejected.

This API requires justified evidence and covariance. Legacy pseudo rows are not automatically promoted into it. An external simulator calibrated with the same observations is not independent; it would need a joint model or explicitly modular treatment. No physical deposition simulator is implemented.

## Validation and volume

The primary workflow partitions unique sites recursively along the longest coordinate span into four approximately balanced spatial blocks. Colocated observations stay together; lexical ordering resolves ties. Nearby but nonidentical locations can lie across a boundary, so `--buffer-km` is available. Empty/insufficient training sets fail visibly. Row permutations do not change the fitted result.

Each candidate refits the trend, scales, amplitude and likelihood inside each training fold. Selection minimizes the average of fold-mean negative log predictive scores on log thickness, including the observation nugget; upper bounds use log-CDF scores and lower bounds use log-survival scores. With `--nested-cv`, outer four-fold prediction uses independent inner three-fold selection. Reported coverage concerns held-out **observations**, while maps show latent-surface uncertainty without the nugget. Neither coverage nor spatial CV validates a wholly unobserved sector.

Since version 0.7, the primary reported fallout volume integrates the entire WGS84 Earth, independently of the user-selected rectangle or map extent. The fit, selected candidates and observation predictions remain unchanged. See [complete-Earth integration](global-volume.md) for the geometry, numerical checks and remaining extrapolation assumptions.

The historical regional simulation is retained separately. For each jointly sampled log surface, finite-domain volume is

\[
V^{(b)}=10^{-5}\sum_j a_j\exp(z_j^{(b)})\quad\mathrm{km}^3,
\]

with cell area a_j in km². Joint draws preserve spatial covariance; pointwise interval bounds are never summed to make a volume interval. Conditional mean thickness is exp(mu+variance/2); median is exp(mu). The default rectangle uses real measurements and vent plus 25% padding. Grid doubling checks the mean integral, not convergence of all sampled quantiles. Domain expansion exposes tail sensitivity; it is not a total-volume estimate.

Runs record input/code hashes, versions, seeds, fold/row assignments, candidates, failures and configuration. Sampling uses a fixed seed; fitting is deterministic. Bitwise identity across numerical libraries/platforms is not promised.

## Limits shown by the experiments

[VALIDATION.md](https://github.com/siyulai21/tephra-fallout-isopachs/blob/41b8f2f28ac8b1c877810b782005ff243e101c63/archive/validation/initial-benchmark.md) reports substantial Aso-4 undercoverage and failure to recover a synthetic hidden lobe. The present code does not establish universally calibrated volume uncertainty. It does establish reproducible fitting without manually invented data and explicit statistical treatment of virtual constraints.

Before publication-level volume claims: jointly infer trend/covariance uncertainty, model independently documented preservation/detection, compare richer anisotropic or multi-lobe means, and evaluate untouched eruptions with preregistered sector holdouts. Those are research extensions, not implemented features of this version.

## Scientific reporting

The Markdown reporting workflow separates the map domain from the finite volume region, expands maps to
resolve closed median level sets, checks enclosed areas (including holes), and
refines mean- and median-surface quadrature. Small contour features receive
local refinement around the globally resolved components. This does not add
pseudo observations, physical inputs, or a new uncertainty model.

See [OUTPUTS.md](../OUTPUTS.md) for numerical definitions, limits, and the
distinction between the median surface integral and median random volume.
[Historical reporting validation](https://github.com/siyulai21/tephra-fallout-isopachs/blob/41b8f2f28ac8b1c877810b782005ff243e101c63/archive/reference/reporting-0.3-validation.md) records the reporting checks.


## Lower bounds and combined-fallout observations (0.6)

A recovered layer that ends before the original base constrains thickness from below. For `Kind=lower_limit` with positive `Limit_cm=L`, the likelihood is

\[
P(Y>L\mid z)=\Phi\!\left(\frac{z-\log L}{\sqrt{\sigma^2+v}}\right).
\]

This is the survival counterpart of the existing upper-limit likelihood, evaluated with a stable log-normal-CDF function. Its negative log is convex in latent log thickness. The same conditional Laplace approximation applies, with the gradient sign reversed and the Hessian form preserved. Trend and covariance scales continue to be fitted to exact observations; bounds constrain the latent field conditional on those estimates. This remains a two-stage conditional method.

All selection, stacking, discrepancy fitting and outer scores use the corresponding tail probability. Interval coverage and RMSE exclude censored rows. Mixing exact-density and censored-probability scores is useful for comparing candidates on the same observation records and censoring pattern; scores across differently censored datasets are not directly comparable evidence of superiority. [Censoring terminology](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.CensoredData.html).

For Minoan, the target is one total field `T_total(s)` for Plinian plus co-ignimbrite/dispersed fallout. At each site, use the reported total minus an identified turbidite, or a recovered-fallout lower bound. Fit and sample the combined field once. No phase boundary is fitted, no missing phase is assigned zero thickness, and no separately fitted fan volumes or interval endpoints are added. A single directional mean with spatial residuals can represent local departures but does not guarantee recovery of every unsampled lobe. [Audit and source decisions](../../data/audits/Minoan/README.md).
