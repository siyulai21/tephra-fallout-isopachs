# Global fallout as a mass-normalized random field

[Project home](../../README.md) · [All-eruption results](../../comparisons/global/README.md) · [Implementation](../../src/empirical_isopach/models/mass_mixture/) · [Support diagnostic](supported-isopachs.md)

Protocol: `mass-normalized-spherical-mixture-v1`, introduced in 0.10. The proposed empirical model estimates **bulk fallout volume**, jointly across accepted fallout fans. It is a research implementation, not a validated physical transport model. Its priors are the same for all eruptions; no wind field, umbrella dynamics, grain-size distribution or published target volume is supplied.

**Version 0.11 adds a source-focused alternative and adopts it for B-Tm after the contour review.** This page documents the preceding displaced geometry, retained for the other five cases; it no longer describes one common geometry for all six published results. See the [source-focused equations](source-anchored-fans.md) and the [B-Tm correction and unresolved Aso-4 comparison](../../comparisons/source-revision/README.md).

## Why the old screen cannot decide whether to give a global estimate

The complete-contour gate was too conservative as a prerequisite for *conditional* global inference. Its own development tests rejected fully sampled secondary-fan deposits. It is still useful for describing where spline/GP contours are stable. A passing contour does not identify an unobserved proximal cap; a failing contour does not forbid probabilistic extrapolation under explicit assumptions.

The new global estimate therefore does not select spline curves, fabricate a closing contour, or integrate whichever GP tail happens to occur outside a data window. It fits a separate positive field whose mass is normalized over the physical Earth. The regional GP and spline comparisons remain peers and useful checks on predictive flexibility. Global estimation no longer depends on their contour gate.

There is a fundamental limit: for finite observations at locations `s_i`, one can add `c b(s)` to any positive field, where `b` is a nonnegative smooth bump supported away from every `s_i`. The likelihood is unchanged while the volume increases by `c ∫b`. Even smoothness and nonnegativity alone cannot produce a data-only finite upper bound. A model class and proper prior are necessary. Here they are explicit, reproducible and varied in sensitivity analyses; they are not claimed to be objective geological facts.

## Geometry and exact mass

Let `R` be the WGS84 authalic radius (about 6371 km), and `s` a unit vector on its equal-area sphere. The source-centred ellipsoidal Lambert equal-area coordinates `(x,y)` are transformed by `(x/D, D y)`, where `D` is the standard latitude-dependent axis scale. This transformation has determinant one. If `p` denotes those coordinates and `r² = |p|²/(4R²)`, then

```text
s = (p/R × sqrt(1-r²), 1-2r²),       0 ≤ r² ≤ 1.
```

Thus solid-angle density divided by `R²` is density per km² of WGS84 surface. The whole domain is the physical Earth, not a radius inferred from the sampling or chosen by an analyst. The coordinate singularity at the antipode has zero area. This geometry does not imply that any particular faraway region contains observable ash.

For component `k`, set a core scale `L_k > 0`, a unit centre `μ_k`, `ℓ_k² = L_k²/R²`, and `u = 1-μ_k·s = |μ_k-s|²/2`. Three radial kernels are used:

```text
G_k(s) = exp(-u/ℓ_k²) / Z_G
Z_G    = 2πR² ℓ_k² [1 - exp(-2/ℓ_k²)]

E_k(s) = exp(-R sqrt(2u)/L_k) / Z_E
Z_E    = 2πL_k² [1-(1+a_k) exp(-a_k)],      a_k = 2R/L_k

P_k(s) = (1 + u/b_k)^(-q) / Z_P,       b_k = q ℓ_k²
Z_P    = 2πR² b_k [(1+2/b_k)^(1-q)-1] / (1-q),   q ≠ 1
Z_P    = 2πR² b_k log(1+2/b_k),                       q = 1.
```

All three integrals equal one because surface area in radial coordinate `u` is `2πR² du`, with `0≤u≤2`. Equivalently, chord distance `r=R sqrt(2u)` has area element `2πr dr`, `0≤r≤2R`, giving the exponential normalizer directly. The factor `b=qℓ²` makes Gaussian and power near-centre curvature comparable; the exponential permits a sharper centre. Stable `expm1`, `log1p` and series expansions avoid cancellation at `q=1` and for broad kernels. The Gaussian component is the established [von Mises–Fisher density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises_fisher.html), expressed per unit area. The other normalizers are derived directly; these kernel ingredients are not claimed as newly invented distributions.

The joint field is

```text
T(s) [cm] = 10^5 V [km³] × Σ_k w_k [ω_G G_k(s) + ω_E E_k(s) + ω_P P_k(s)],
w_k ≥ 0,   Σ_k w_k = 1,   ω_j ≥ 0,   Σ_j ω_j = 1.
```

Consequently `10^-5 ∫_Earth T(s) dA = V` **for every posterior draw**. Integrating the pointwise median map is not used to estimate median volume. There is no cell-size correction, contour cutoff, map-edge closure, fitted calculation radius or arbitrary finite rectangle in this identity.

Each component decreases away from its own centre. The *sum* need not decrease in every direction from the vent: displaced and overlapping components permit multiple fans, elongated deposits and secondary maxima. Baseline components are circular on the authalic sphere; a few overlapping components approximate anisotropy but cannot represent every real deposit. Mathematical components must not be relabelled as separately identified Plinian/co-ignimbrite phases. Minoan is fitted once to all its accepted fallout observations and lower bounds.

## Fixed general priors

All distributions below use natural logs and standard deviations. These numbers encode modelling judgment, not fabricated measurements. They were frozen for the final all-eruption rerun after the development checks described below. No eruption-specific settings or published target volumes enter them.

| Quantity | Baseline prior | Reason and limitation |
|---|---|---|
| Fan count | `K=3` | Modest multi-fan flexibility; checked against 1 and 5 |
| Total bulk volume | `log V ~ Normal(0,6)` in km³ | Proper broad prior, approximately `8×10^-6`–`1.3×10^5` km³ central 95%; not “uninformative” |
| Each core scale | `log L_k ~ Normal(log 300,1.5)` in km | Common range spanning small to continent-scale deposits; scale sensitivity checked |
| Centre displacement | Each coordinate of `z_k ~ Normal(0,3)` | A soft relation of displacement to fan scale, with no preferred azimuth |
| Fan mass weights | `w ~ Dirichlet(0.3/K,…,0.3/K)` | Sparse prior allows unnecessary fans to carry negligible mass; fixed total concentration across fan-count sensitivities |
| Kernel mass weights | `ω ~ Dirichlet(0.1,0.1,0.1)` | Shared across fans; permits approximately pure exponential, Gaussian or power thinning, as well as mixtures |
| Tail exponent | `q = 1 + exp(h)`, `h ~ Normal(0,1)` | Tails would be integrable on an infinite plane; a sensitivity relaxes that constraint |
| Residual log scatter | `σ ~ HalfNormal(1.5)` | Estimated from observations, not added to deposit mass |
| Broad-error fraction | `ε ~ Beta(1,9)` | Allows occasional large deviations; not a declaration that any record is erroneous |

Centres use `a_k=L_k z_k/(2R)` and `μ_k=(2a_x,2a_y,1-|a|²)/(1+|a|²)`. This smooth stereographic map avoids longitude seams. There is no hard distance exclusion. Neither source smoothness nor finite mass rules out hidden deposition, so these priors can substantially determine the global upper tail.

Sparse mixture weights are motivated by the established idea of shrinking redundant components, discussed by [Rousseau & Mengersen (2011)](https://doi.org/10.1111/j.1467-9868.2011.00781.x). Their asymptotic results concern density-mixture sampling and are **not** a convergence or coverage theorem for this spatial thickness-regression model. The concentration 0.3 is a declared development choice; the concentration-3 refit exposes its influence rather than presenting it as universal volcanology.

## Observations and bounds

At each accepted location, the exact log-thickness `y_i` has mixture likelihood

```text
(1-ε) Normal(log T(s_i), σ² + known_sd_i²)
   + ε Normal(log T(s_i), 16σ² + known_sd_i²).
```

For an upper limit use the corresponding mixture CDF; for a lower bound use its survival probability. Known measurement errors enter once, without multiplying them by the broad-error factor. Bounds are not exact thicknesses and absent samples are not zeros. Legacy pseudo rows are ignored, including their coordinates. No pseudo-observations, fitted contours, sampled integration points or predicted values are fed back into the likelihood.

The field is a latent central thickness under multiplicative observation/preservation scatter. It is **not** multiplied by `exp(σ²/2)` to inflate total volume: residual scatter belongs to the observation model. Systematic erosion, preferential sampling, coring bias, reworking, correlation between records and incorrect source assignments can still bias inference. These require evidence and potentially a different likelihood, not invented uncertainty corrections.

This distinction is substantive: if the residual scatter represents real unresolved *areal deposition variability*, rather than observation/preservation error, integrating the central field need not recover the area-mean physical mass. An identifiable spatial-heterogeneity model or independent sampling information would then be needed. The present credible interval does not automatically include that structural correction.

## Uncertainty, sensitivity and nonidentifiability

Four independent adaptive Hamiltonian chains sample the joint posterior with [NumPyro NUTS](https://num.pyro.ai/en/stable/mcmc.html). Internally define `g(θ)` as the mean log thickness for unit volume at the *training locations*, and sample `a = log V + g(θ)`. Then `log V = a-g(θ)` and the log field is `a+log shape-g(θ)`. The transformation has Jacobian determinant one. An explicit density-ratio factor cancels the auxiliary normal on `a` and restores the declared prior on `log V` exactly. This reduces amplitude/shape coupling without changing the scientific posterior or using heldout values. Unit tests compare the complete joint density, including bounded observations and the shifted-volume prior.

The baseline median and central 95% interval are Bayesian posterior summaries **conditional on the stated family, likelihood and priors**. Full parameter uncertainty and the uncertain mass weights among fans are included. Finite mixtures can be multimodal; passing numerical checks does not prove all modes were explored.

All seven alternatives are full posterior refits to the same accepted data:

1. **Diffuse tails:** `q=exp(Normal(log 2,1))`, allowing `q≤1`, still finite on Earth.
2. **Broader geography:** scale log-SD 2.25 and centre-offset SD 6.
3. **Volume-prior shift:** increase the prior median volume 100-fold, retaining log-SD 6.
4. **One fan:** same total Dirichlet concentration convention and other priors.
5. **Five fans:** same convention and other priors.
6. **Gaussian errors:** remove the broad-error component.
7. **Diffuse mixture weights:** total Dirichlet concentration 3 for both fan and kernel weights, instead of 0.3. This tests how strongly the estimate depends on allowing unused components to become negligible.

Each has its own conditional interval. The minimum lower endpoint and maximum upper endpoint form an **assumption-sensitivity envelope**, not a 95% interval or model average. This finite list is not an exhaustive set of plausible deposits. Models are not chosen by whether they reproduce a desired published volume. A materially different result is information about assumption dependence, not a reason to hide the alternative.

Reports also show in-sample posterior predictive log-RMSE for every alternative. This can reveal a restrictive shape, but it is optimistic, is not independent validation, and does not supply model probabilities. Only the baseline is evaluated on the complete matched geographic holdout protocol.

A restrictive alternative can trade an unsuitable shape against a vastly larger unseen deposit, even while its sampler converges. The one-fan Aso-4 result is an example. Such geologically extreme outcomes remain visible as failures of identification or model adequacy; the envelope does not certify them as plausible eruption volumes. No upper truncation is inserted to make that diagnostic look more reasonable.

The mass partition reports the inferred fraction within the existing geometry-screened footprint, in an unobserved proximal cap, beyond the most distant exact site, and in other unsampled gaps/sectors. These disjoint classes describe sampling geometry, not geological classes or validated mass. Positive Sobol quadrature uses each kernel's exact radial inverse CDF, checks doubling error, and never sets the global total. Posterior quantiles of parts need not sum to posterior quantiles of the total.

## Numerical and empirical checks

Rank-normalized split R-hat and bulk/tail effective sample sizes follow [Vehtari et al. (2021)](https://arxiv.org/abs/1903.08008). Checks use volume, tail/noise parameters, the observation-mixture probability, weighted scales, effective component count, predictions at observed sites and probes in unsampled sectors. The observation-mixture check is recomputed from saved samples when loading verified caches; immutable attempt records retain their original sampler diagnostics, with the complete posterior assessment added in reports. Raw exchangeable component labels are not used to declare convergence. The pass rule is maximum R-hat below 1.01, minimum bulk and tail ESS at least 400, zero divergences, and energy BFMI at least 0.3 in every chain. Log-volume quantile Monte Carlo standard errors and per-chain volume quantiles are also retained. Maximum-depth saturation of at least 1% is a separate efficiency flag. [Stan's diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html#maximum-treedepth) distinguishes this from a validity failure when other diagnostics are satisfactory. Initial development attempts also applied a depth-based veto; their original flags remain in the development audit. This changes numerical effort, not the posterior or scientific priors.

Initial effort is 1,000 warmup and 1,500 retained draws per chain, target acceptance 0.95, maximum depth 11, seed 20261104. If diagnostics fail, two independent retries use 2,000 warmup and 3,000/6,000 retained draws per chain, target 0.98 and recorded seeds. Use the final attempt, retaining the earlier diagnostics; never select or trim individual chains to make checks pass. A remaining failure is prominently marked as numerically provisional. Extra numerical effort does not change the scientific priors.

Geographic validation uses exactly the regional comparison's four outer partitions and geometry-only buffer. Every global posterior is refitted on its training data; there is no response-selected global tuning. Scores integrate the complete posterior observation mixture, including documented bound events. Better withheld predictions can support local transferability but cannot establish global-volume accuracy.

The first development model used Gaussian/power kernels and diffuse weights. On one smooth exponential-thinning deposit with true volume 33.57 km³ it gave a median 82.72 and interval 43.07–1,679.96 km³, missing the truth despite a broad upper tail. Adding an exponential kernel alone still gave 79.35 (39.55–2,175.99) km³. This exposed both a shape limitation and the prior's tendency to allocate unseen mass to extra components. The final development revision adds the exponential kernel and sparse Dirichlet weights; the diffuse-weight alternative remains a full sensitivity refit. These revisions used a known-truth development case and are **not** independent validation. Earlier attempts and their source hashes are preserved in the development record.

The final fixed suite uses fresh seeds and includes smooth thinning, multiple fans, missing sectors, an invisible proximal cap, a matched spherical field and a diffuse global tail. The hidden cap and diffuse-tail cases deliberately challenge the baseline assumptions. Remaining coverage failures are retained. This small development suite is not simulation-based calibration or external validation. Existing matched GP/spline tests are rerun as a regression and regional comparison check.

The empirical-volume literature already recognizes dependence on thinning assumptions and exposure, and motivates comparing alternative extrapolations; see [Bonadonna & Costa (2012)](https://archive-ouverte.unige.ch/unige:19461). It does **not** validate the particular priors or mixture model proposed here. The contribution of this implementation is the combination of direct whole-Earth mass parameterization, bounded/robust observation likelihood, full posterior sampling, common generic assumptions, sensitivity refits and auditable global reports.

## Reproduction and files

```sh
tephra-isopachs global YTT --output results/ytt-global
tephra-isopachs global YTT --output results/ytt-global --resume
python -m benchmarks.mass_mixture --output results/global-stress
```

The first command starts a fresh run; resuming requires matching input, prior, model-source, Earth-geometry source, runtime versions, numerical settings and posterior checksums. Each worker freezes its source/runtime identity when loading the model so a concurrent file edit cannot relabel its later fits. Reports preserve every attempt's diagnostics and source identity. Baseline complete-field posterior samples and all sensitivity volume samples are published; other full-field samples remain in the verified local run cache and can be regenerated from the recorded inputs/settings. NumPyro/JAX version and platform differences can change individual draws; they should not be confused with a change in the scientific target distribution.
