# Supported isopachs, regularization and unresolved volume

This protocol governs local support and contour-band continuation. The [probabilistic global model](probabilistic-global.md) supplies conditional whole-deposit estimates independently of its eligibility gate.

[Method catalogue](../../methods/README.md) · [Regional protocol](regional-comparison.md) · [Implementation](../../src/empirical_isopach/contour_support.py)

**Smoothness, closure and support are different properties.** A spline may oscillate outside well-sampled ground. A GP may return smoothly to its mean there. Neither behavior supplies missing observations. Version 0.9 adds a shared support audit to both methods; it does not choose a winning family, impose a preferred eruption volume, or manufacture points to close a contour.

This is a reproducible **development protocol**, not a calibrated contour-confidence procedure or a claim to have invented a new class of statistical models. Its geometric filtering, geographic deletion and area-based volume identities have familiar mathematical foundations. The contribution here is their explicit combination, failure rules, implementation and evaluation for this workflow.

## General volcanological judgments

We require positive thickness and keep bulk fallout distinct from ignimbrite, reworked deposits and dense-rock equivalents. Missing, eroded or unvisited sites do not establish zero deposition. A documented absence requires a detection limit; an incomplete core can supply a lower bound, not an exact recovered total.

Multiple fans, secondary thickness maxima, islands and holes remain possible. We do **not** require every isopach to be an ellipse, star-shaped around the vent, or monotonically thinning along every ray. The directional background trends are regularization assumptions; the fitted residual surfaces may depart from them. Minoan Plinian and co-ignimbrite observations remain one total-fallout field. All components of each fitted superlevel set enter the area calculation together.

The well-measured middle of a deposit, its unmeasured proximal core and its distal tail are separate inferential problems. [Klawonn et al. (2014)](https://pubs.usgs.gov/publication/70170796) found much larger uncertainty in proximal and distal extrapolation than in their well-sampled intermediate deposit. Their numerical fractions are specific to that case and are **not** transferred here as universal priors. [Yang and Jenkins (2023)](https://link.springer.com/article/10.1007/s00445-023-01652-1) likewise distinguish interpolation/model uncertainty from extrapolation and advocate reporting these portions separately.

## Pseudo-data: replace invented observations with explicit penalties

The active spline already minimizes the real-observation negative log likelihood plus integrated derivative and trend penalties. For the trend term,

\[
\frac{\lambda}{2s^2|D|}\int_D [f(x)-m(x)]^2\,dx
\approx \frac{\lambda}{2s^2}\sum_q w_q[f(x_q)-m(x_q)]^2,
\qquad \sum_qw_q=1.
\]

The quadrature locations are integration nodes, **not pseudo-measurements**. Duplicating them with split weights leaves the penalty unchanged. Coordinates, thicknesses and variances of an invented closing ring never enter the likelihood. The trend target is estimated from the same training data: this is data-adaptive regularization, not independent prior evidence. Trend strength, including zero, and spline settings are chosen within geographic training folds. See the [spline equations](cubic-splines.md).

The GP's mean and kernel serve the same general purpose of structuring predictions where evidence is scarce. They are not extra observations either. Reusing a prediction from either model as independent pseudo-data would count the original information twice. Genuine independent evidence may enter with its documented likelihood and dependence; a copied isopach cannot.

Consequently, **we do not choose pseudo-data to make every eruption close**. If an eruption lacks closure evidence, the correct output can be a partial reconstruction and an unresolved global volume. General volcanology alone does not identify a distant thickness, a maximum proximal thickness, or a missing fan's area.

## A common local interpolation footprint

The observation hull remains a matched historical comparison region. It can connect distant clusters through almost unvisited space. The new footprint is the union of accepted Delaunay triangles of **distinct exact measurement locations**; the vent, legacy pseudo-points and bounds are not inserted as exact thickness sites. Bounds continue to inform the fitted likelihood.

For each vertex, let s be its distance to the third nearest distinct exact site. Retain a triangle only when both its longest edge divided by the minimum vertex s and its circumdiameter divided by the median vertex s are at most a fixed factor. The first rejects long bridges; the second also rejects long thin triangles. Duplicating a measurement, changing thickness values, changing coordinate units or changing the display rectangle cannot enlarge this geometry. Holes and disconnected patches are retained.

| Policy | Spacing factor | Maximum deletion shape change |
|---|---:|---:|
| Strict | 1.5 | 10% |
| Standard | 2 | 25% |
| Permissive | 3 | 50% |

These values and the third-neighbor rule are **declared algorithmic judgments**, not volcanological constants or estimated confidence levels. Every eruption gets every policy. Nothing is relaxed after seeing a desired volume. A universally assumption-free support boundary is not identifiable from point samples, particularly under preferential sampling. Local support can still be misleading when all available observations share preservation bias or an unrecognized deposit correlation error.

Both methods integrate exactly the same triangles with positive quadrature. Expected thickness and median thickness are integrated separately; the latter is not the median of the random volume. Two successive refinements below 0.1% are required. Failed numerical checks remain visible. Four fully retuned geographic-deletion models are integrated on the **same fixed footprint**, exposing fitting sensitivity separately from footprint sensitivity. Neither range is a confidence interval. These partial volumes are not certified lower bounds on true fallout.

## Which complete contours qualify?

The candidate grid has 25 equally spaced log-thickness values between the exact-observation 5th and 95th percentiles; 49 levels test grid-density sensitivity. Percentiles avoid letting a single extreme set the entire display scale. They are not probability statements about the deposit.

For each method and each level, require:

1. The **entire superlevel set** closes inside the numerical rectangle; no open ends are joined. If any boundary remains above the level, the complete area is unresolved. All islands and holes count.
2. The entire contour perimeter lies inside the local footprint, apart from numerical geometric tolerance. Report unsupported enclosed area separately: a reliable ring may surround an unmeasured proximal core.
3. Each above-level component contains at least three distinct exact sites reported above the level; at least three exact sites outside report below it. Every hole also needs three below-level sites. These noisy-data bracketing checks are diagnostics, not proof that latent thickness lies on a particular side.
4. Coarse/fine contour symmetric difference, divided by their union area, is at most 1%. This checks numerical geometry, not geological accuracy.
5. The contour still closes after **each** of four geographic deletions with complete model retuning. Each deletion's symmetric-difference/union area is below the policy's tolerance. Stable total area alone is insufficient: a displaced contour can have unchanged area.

Rejecting a curve does not reject its surrounding measurements. Integrating the common local footprint remains possible without any complete isopach.

A usable thickness band needs at least three consecutive qualifying contours, at least a factor of two in thickness, and no unsupported area anywhere between consecutive contours. Every intermediate level on the refined 49-level grid must also pass; a band cannot qualify just by omitting an inconvenient intermediate contour. The widest log-thickness span is selected, with contour count resolving ties; every run remains in the record. The proximal core is **not** included just because an outer perimeter qualifies. This conservative policy can reject genuinely correct contours; abstention is preferable to calling a model-generated closure observed. Tests quantify behavior on fixed synthetic cases, not universal false-acceptance probabilities.

## Integrating the band without choosing an arbitrary subset of spline curves

For a positive thickness surface T, define A(t) as the total area where T >= t. These superlevel sets are automatically nested even for multiple fans and irregular contours. The layer-cake identity is

\[
V=\int_0^\infty A(t)\,dt.
\]

But for the **geographic band** between an outer thickness l and an inner thickness h,

\[
V_{\rm band}=lA(l)-hA(h)+\int_l^h A(t)\,dt.
\]

Omitting the boundary terms integrates a vertical slab, a different quantity. We integrate the fitted surface directly over the selected polygonal band, preserving holes using constrained triangulation, and check it against lower/upper Riemann sums of the decreasing A(t). Those bounds are numerical integration checks conditional on fitted areas, not confidence limits. The endpoints define the band; additional contours improve quadrature and **never increase statistical sample size**.

The selected band depends on responses and fitted surfaces. No selection-adjusted band-volume interval is claimed. Conditional field uncertainty, model-selection uncertainty, contour-selection uncertainty, numerical error and missing-deposit uncertainty must not be conflated. A future calibrated excursion-set method would require additional assumptions and validation; pointwise GP intervals do not supply simultaneous contour coverage.

## Extending an eligible band: scenarios, never an automatic preferred total

Only after the gate passes do we form an area-thickness continuation, using x = sqrt(A). The band keeps its direct median-surface integral. Endpoint secants over one third of the log-thickness span define proximal and distal slopes, so increasing contour density does not supply fictitious independent evidence.

Two proximal assumptions are shown: a constant cap at the highest reliable thickness, and an exponential continuation of the inner slope. Two distal assumptions are shown: an exponential continuation and a power continuation matching the outer value and slope. All are stated assumptions; none is a recommended total. The power tail is integrable over infinite area only when its exponent exceeds two. Otherwise it is explicitly reported as nonintegrable, without an arbitrary outer cutoff. These are infinite-area empirical volume scenarios, not a geographic reconstruction of the opposite hemisphere or validated complete-Earth volumes.

There is an unavoidable identifiability limit. Beyond the outer reliable level, tails T(x)=l(x/x_l)^(-(2+epsilon)) all match its endpoint and have finite volume for epsilon>0. Their distal volumes are 2*l*x_l²/epsilon, which have **no finite upper bound** as epsilon approaches zero. A smooth transition can hide the slope change beyond the sampled region. An unmeasured proximal patch can likewise contain an arbitrarily large smooth positive cap without altering any observation. Requiring positivity, smoothness and finite mass cannot resolve either problem.

Therefore no preferred global estimate or global confidence interval is emitted by this audit. Even the finite scenario range is not an exhaustive uncertainty range. When no band qualifies, no continuation is fitted. This is a substantive scientific outcome, not an invitation to add closing pseudo-points until a number appears.

On a finite Earth, an additional assumption that every unsampled distal location is thinner than l gives a finite area-times-thickness upper bound. It is generally very weak and depends on ruling out unobserved fans; the measurements do not establish that assumption. The unmeasured proximal-cap ambiguity remains even on a finite Earth. The infinite-area power example describes the empirical continuation convention, not an infinite physical planet.

## The fitting rectangle is a different sensitivity

Changing a map window cannot change the data-defined footprint or its integral for a fixed fitted field. Changing the spline's **fitting** rectangle can change its basis and normalized penalties, even inside that footprint. We explicitly refit and retune at 2.5%, 5%, 10% and 20% padding while integrating the same standard footprint. This is sensitivity analysis, not a new outer-test comparison or a mechanism to choose a desired volume. GP fitting does not use this rectangle. The 5% spline rule remains the common baseline; remaining sensitivity is reported rather than described as eliminated. See the [all-eruption audit](../../comparisons/support/README.md).

## Evaluation and reproduction

`tephra-isopachs compare` now produces the support audit as part of each normal upload/run/report workflow. The original method tuning and matched-hull results remain available for comparison. Four outer folds are evaluated separately inside and outside **training-only** support; withheld coordinates are not added to that mesh. Counts and row identities accompany the metrics, so empty or tiny supported strata cannot masquerade as calibrated performance.

```sh
python -m benchmarks.regional --output results/new-regional-evaluation
python -m benchmarks.support --output results/new-support-stress-tests
python -m benchmarks.support_domain --base results/new-regional-evaluation --output results/new-domain-sensitivity
python -m pytest -q
```

The standard regional benchmark retains its 40 attempted trials. Additional support trials use smooth deposits, a secondary fan, a missing sector and an unobserved proximal cap, with full nested tuning of both families. Analytic tests independently verify holes, multiple islands, open-contour rejection, area-thickness integration and finite/nonintegrable tail behavior. Source hashes, policies, rejected levels and numerical failures are retained. A fixed known-truth development suite cannot certify performance for all eruptions.

[Engwell et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4498447/) provide direct precedent for distinguishing closed spline isopachs from unsupported boundaries and studying contour stability under subsampling. Their empirical sample-size findings are not universal thresholds and are not used as such here.
