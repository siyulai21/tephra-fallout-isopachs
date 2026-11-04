# Fallout volume without a user-selected calculation region

The current peer-method workflow uses [matched regional volumes](regional-comparison.md). This page describes the retained complete-Earth GP calculation.

[Method overview](../METHODS.md) · [Output definitions](../OUTPUTS.md) · [Historical GP comparison](../../reports/global-fallout/README.md)

Version 0.7 makes the **complete-Earth integral** the primary fallout-volume estimate. `--bounds` controls only a regional diagnostic and prediction export. Map extent and contour thresholds cannot change the global estimate. The accepted observations, likelihoods, selected models and nested validation are unchanged by this integration revision.

## A physical domain

A deposit occupies the Earth, including land and ocean. The complete WGS84 surface is a fixed physical domain. An arbitrary rectangle can omit deposit; an infinite flat plane extends beyond that domain. A mathematically integrable power tail can still put enormous volume at physically meaningless distances when its decay exponent approaches two. Increasing the power-law exponent floor would introduce an extra tail assumption rather than solve the geometry.

The existing model uses WGS84 Lambert azimuthal equal-area (LAEA) coordinates. The global image is an ellipse whose axes follow the ellipsoid and source latitude. The antipodal point is a boundary singularity of zero area; quadrature uses interior points. Ellipsoidal LAEA is defined globally in [PROJ](https://proj.org/en/stable/operations/projections/laea.html). Its conversion formulas are specified in [IOGP Guidance Note 7-2, section 3.3.2](https://www.iogp.org/wp-content/uploads/2019/09/373-07-02.pdf) and [Snyder's USGS projection manual, chapter 24](https://pubs.usgs.gov/pp/1395/report.pdf).

With authalic radius Rq and LAEA axis scale D, write

\[
x=D\rho\cos\theta,\qquad y=D^{-1}\rho\sin\theta,
\quad 0\leq\rho\leq2R_q,\quad 0\leq\theta<2\pi.
\]

The Jacobian is exactly rho, so the total area is 4πRq², equal to the WGS84 ellipsoid area. The code computes Rq and D from WGS84 flattening and source latitude; it does not substitute a circular spherical boundary for the ellipsoidal projection.

For the conditional latent log field with mean mu and variance v,

\[
E[V_E\mid y,\widehat\theta]
=10^{-5}\int_0^{2\pi}\int_0^{2R_q}
\exp\{\mu(x,y)+v(x,y)/2\}\rho\,d\rho\,d\theta.
\]

The conversion is from cm × km² to km³. This is **bulk fallout**, with no automatic density/porosity correction. In a stacked model, integrate the weighted component lognormal expectations. The global median-surface integral uses the exact marginal mixture median and is a separate quantity, not the median of random total volume.

This is a new integration and reporting procedure for this repository, using established projection and quadrature mathematics. It is not a claim of a new geodetic theory or a validated new physical dispersal model.

## Quadrature and checks

Positive Gauss-Legendre weights integrate the transformed radius `t=log(1+rho/s)`. The scale s is the median positive observed rho and affects numerical point placement only. A periodic angular rule covers all directions. Observed-distance quartiles, the farthest observation and the hemisphere boundary subdivide integration; their union always covers the complete Earth. No annulus is discarded for being thin or distant.

The radial/angular orders are (24,64), (48,128), (96,256), (192,512), (384,1024). Two consecutive refinements must agree within 0.1% for both expected and median-surface volume. Annulus-wise absolute changes, normalized by each total, must also pass, preventing cancellation between annuli from hiding spatial redistribution. A shifted angular grid must then agree to the same tolerance. Failure remains explicit in JSON and Markdown. These are empirical convergence checks, not rigorous error bounds; [SciPy's Gauss-Legendre implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html) supplies nodes and weights.

Independent tests cover WGS84 area, uniform-thickness volume, closed-form radial exponential and critical-power integrals, agreement with PROJ at equatorial/oblique/polar origins, a separate geodetic area integral, exact mixture expectations, numerical failure reporting and changing CLI rectangles without changing the global result. Known synthetic deposits test recovery separately from quadrature accuracy.

## What the diagnostics mean

- **Beyond the farthest observation:** expected volume at source distances greater than every input location. Distances are arcs on the equal-area authalic sphere, not exact WGS84 geodesics. This is a conservative distance-based extrapolation indicator; unsampled directions and gaps also exist inside that distance.
- **Opposite hemisphere:** expected volume beyond an authalic central angle of 90 degrees. A large value signals particularly long-range dependence on the projected tail model.
- **Other thinning family:** the best validated candidate of the other family on the same input, integrated over exactly the same physical domain. This is a sensitivity comparison, not independent evidence or a confidence interval. For an ensemble the alternative may already be a component.
- **Geographic deletion:** remove an outer geographic group and its validation buffer, refit using the candidate selected inside that training set, then reintegrate globally. The four resulting volumes measure sensitivity to available spatial evidence; their range is not a statistical interval.
- **Infinite-plane background:** an analytic stress diagnostic for the unconditioned trend plus marginal GP variance. It omits the conditioned spatial correction and extends beyond Earth, so it is not reported as physical fallout volume.

For the background diagnostic, if `q(theta)=b-cx*cos(theta)-cy*sin(theta)` and `h=r/r0`, the radial integral is `r0²/q²`. If `h=log(1+r/r0)`, it is `r0²/((q-1)*(q-2))` when q>2. Analytic angular integration exposes near-critical tails without pretending that an ever-larger finite rectangle has captured them.

## Remaining statistical assumptions

The new result has no arbitrary calculation-region parameter. **It still depends on the extrapolation model.** A smooth positive field in an unsampled region can change true total volume without changing observations. Complete-Earth integration cannot identify such missing information.

The existing planar directional trend and Matérn covariance are extended through the global LAEA projection. They are not a spherical atmospheric transport model. Far-field projection distortion, the antipodal representation, positive expected thickness over unsampled areas, preservation, source position and unknown deposit structure remain limitations. Numerical stability is not evidence that those global assumptions are correct. The comparisons intentionally show large disagreements rather than selecting a tail because it matches a preferred published volume.

**No global probability interval is estimated.** Conditional simulations in `report.json.volume` concern the separate historical rectangle. They must never be relabelled as global intervals. Fitted-parameter and model-selection uncertainty remain omitted. Other eruption products and DRE require additional evidence.

## Reproduce

```sh
python -m benchmarks.global_fallout --output results/global-fallout
tephra-isopachs fit Minoan --output results/minoan --nested-cv
python -m pytest -q
```

The first command reruns both methods for all six catalogued eruptions, 24 geographic-deletion integrals and 30 known-truth synthetic fits. It preserves the latest accepted YTT principal selection and fits all accepted Minoan fallout fans together. Per-run source/input hashes and the benchmark hash identify the calculation. [All-eruption results](../../reports/global-fallout/README.md).
