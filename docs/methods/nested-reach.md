# Historical Bayesian nested isopachs with dispersal corridors

The station-corridor formulation is retained for reproduction and paired field tests. Current analyses use [nested dispersal footprints](nested-footprints.md).

A distant ash bed establishes thickness at its location. It supplies limited information about a distant site on another bearing, even when both locations have the same radius in an idealized isopach. The geographic reach fit retains the nested thinning model, its source and detached-fan alternatives, and its analogue-informed distal continuation. It adds uncertainty about how far measured bearings can be continued across wind and beyond the last finding. The companion whole-area fit shows how much this choice matters. [Paired estimates](../../comparisons/nested-reach/README.md) · [Tests against actual surveys](../../comparisons/nested-reach/field-surveys/README.md) · [Original thinning and shape model](nested-isopachs.md)

## Construction

The source-centred, equal-area Earth chart gives each positive site a chord radius and compass bearing. An exact thickness, a lower bound or a positive interval supplies a potential source-to-site corridor. The median distance of the three closest distinct positive locations sets the proximal spatial resolution. With fewer than three, the median uses the available sites. A finite upper limit contributes its probability to the fit. Occurrence-only reports without a quantitative bound have no thickness likelihood.

For a prediction point and a positive site at radius $R_i$, let $a$ and $b$ be the along-bearing and perpendicular coordinates in kilometres. Write $u=\min(R_i,\max(0,a))$ and let $r_0$ be the proximal resolution. The squared scaled distance from its corridor is

$$
d_i^2 =
\frac{b^2+\min(a,0)^2}{[(r_0+u)\tan(w)]^2}
+\frac{\max(a-R_i,0)^2}{[e(r_0+R_i)]^2}.
$$

For each complete nested deposit draw, the geographic multiplier is

$$
M = [1+(\sqrt{2}-1)\min_i d_i^2]^{-2}.
$$

The multiplier equals one along a positive source-to-site segment, one half at unit scaled displacement, and stays positive beyond the survey. Adjacent corridors can overlap. Overlap preserves the largest reach without adding repeated deposits. A possible source-to-site transport path is a modeling assumption. The measured part of a contour requires separate local bracketing.

Each finite-reach draw uses one half-width $w \in \{12°,30°,60°\}$ and one down-range extension $e \in \{0.25,0.5,1\}$. The nine combinations share 75% of prior probability and $M=1$ has 25%. These are transferable structural priors. No target volume or eruption-specific wind direction enters them. Case reports vary the broad probability from 10% to 75%. Where all training records are positive findings, the observed thicknesses inform the parent thinning field but cannot identify this multiplier away from those sites.

The analogue-based thickness continuation joins at the greatest parent area rank of a positive finding. Upper-only locations can therefore influence posterior shape and width without extending the positive reach. Multiplying a complete thickness draw preserves nesting of its superlevel sets. A conditional model can still have a detached region or a long, unmeasured lobe.

## Fitting and integration

The saved whole-area posterior provides an importance proposal. At every positive training location the corridor model and the intrinsic parent have identical likelihood. For upper limits the update divides out the original observation likelihood and replaces it with the event probability under the new field. If any independent population has effective sample size below 64, relative effective size below 2%, or a maximum normalized weight of at least 5%, the pipeline refits the intrinsic posterior on positive findings alone. The upper-limit likelihood then enters once. Every attempt is recorded.

The final update marginalizes over all ten reach laws for every parent particle. Independent randomized draws within strata preserve variation in law when parent particles have equal weight. The pipeline checks four independent populations. Ratios between their largest and smallest volume quantiles must be below 1.35 at the median and 2 at each 95% endpoint. Finite-Earth volume integrates thickness in centimetres over area in square kilometres with a factor of $10^{-5}$. Two scrambled equal-area quadratures, refinement, and the analytical parent integral check numerical error. Failed parent sampling, updated volume checks and integration checks remain separate. [Field implementation](../../src/empirical_isopach/models/nested/reach.py) · [Whole-Earth integration](../../src/empirical_isopach/models/nested/reach_integrals.py)

## Reading the maps

The central map is the pointwise median field. The global volume estimate integrates that same median-thickness field over Earth. The conditional 95% interval comes from the lower and upper endpoints of the spatially correlated volume draws. The local map marks a contour solid only where a retained triangle has exact observations on both sides at distinct vertices and at least 90% of posterior draws cross that level between its vertices. The per-triangle file records each decision. The full-compass view includes the unsampled flanks and a conditional 10–90% exceedance belt. The measured footprint, local bracket and full-contour support screen answer different questions. [Shared support protocol](supported-isopachs.md)

The corridor supplies the principal nested map, volume and held-out score for every eruption. The whole-area field supplies the paired sensitivity. Use held measurements, censoring bounds and independently surveyed deposits to judge geographic reliability. The Calbuco tests use a campaign and a mapped distal contour that stayed outside the fit. Simulation provides analytic totals and known missing lobes for testing behavior under specified conditions. It supplies no frequency estimate for real eruption dynamics. [Independent surveys](../../comparisons/nested-reach/field-surveys/README.md) · [Known-surface checks](../../comparisons/nested-reach/known-surfaces.csv)

## Reproduction

The default full comparison generates both nested fields. A stand-alone run can also use the geographic model directly.

```sh
tephra-isopachs global Aso-4 --geometry nested-isopachs --output results/aso-corridor
tephra-isopachs global Aso-4 --geometry nested-whole-area --output results/aso-whole-area
python -m benchmarks.nested_reach --output results/nested-reach-cases --workers 2
python -m benchmarks.nested_reach_challenge --design quadrants --output results/nested-reach-quadrants --workers 2
python -m benchmarks.nested_reach_challenge --design known --output results/nested-reach-known --workers 2
python -m benchmarks.nested_reach_field --output results/nested-reach-field --workers 2
python -m benchmarks.nested_reach_field_summary --field results/nested-reach-field
python -m benchmarks.verify_nested_reach
```

[Case navigation before the paired study](../../archive/2026-11-21-nested-reach/README.md)
