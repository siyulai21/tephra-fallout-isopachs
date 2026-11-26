# Directional geography and local thickness

The principal directional method now separates broad dispersal from local thickness variation at a fixed physical scale. Its source harmonics have smooth Cartesian curvature, its basis can resolve dense surveys, and its distal attenuation is inferred from thickness rather than survey reach. Every plotted line comes from the fitted posterior; no smoothing or removal of contours follows inference. [Equations and numerical procedure](../../../docs/methods/directional-multiscale.md)

The matched comparison uses all nine accepted inputs, unchanged from revision [beee82b](https://github.com/siyulai21/tephra-fallout-isopachs/tree/beee82b7020d62aa2c3dc847d74e45dcf651084f). There are 1,091 accepted rows: 883 point readings, 61 ranges and 147 upper limits. Reported minima enter as points at their stated values at every distance. Upper limits and intervals retain their original event probabilities. Unsurveyed directions provide no absence observations. The [archived v5 packets](../../../archive/2026-11-25-directional-v5/README.md) use these same inputs; the older [multiscale study](../multiscale/README.md) uses an earlier observation policy and is a separate historical comparison.

## Agreement and geographic transfer

Mean absolute log error at measured sites falls from 0.352 to 0.311 when eruptions receive equal weight, an 11.5% reduction, with improvement in every case. Pooled over point readings, it falls from 0.374 to 0.343; the fraction within a factor of two rises from 83.7% to 87.0%. Equal-eruption geographic negative log score improves from 1.376 to 1.354 and quadrant score from 1.570 to 1.556. Lower values are better. These are modest average transfer gains, not improvement on every deletion.

<!-- matched-results -->

| Eruption | Site error v5 → v6 | Geographic score v5 → v6 | Quadrant score v5 → v6 |
| --- | ---: | ---: | ---: |
| aso-4 | 0.404 → 0.330 | 1.321 → 1.225 | 1.552 → 1.429 |
| b-tm | 0.334 → 0.252 | 1.397 → 1.414 | 2.390 → 2.405 |
| ilopango-tbj | 0.679 → 0.656 | 1.808 → 1.734 | 1.664 → 1.642 |
| mazama | 0.303 → 0.284 | 1.339 → 1.380 | 1.239 → 1.423 |
| minoan | 0.491 → 0.464 | 1.603 → 1.597 | 1.670 → 1.670 |
| pinatubo-1991-layer-c | 0.207 → 0.199 | 0.949 → 0.889 | 0.897 → 0.861 |
| sakurajima-satsuma | 0.328 → 0.310 | 1.052 → 0.993 | 1.319 → 1.289 |
| tambora | 0.237 → 0.156 | 1.500 → 1.437 | 1.545 → 1.471 |
| ytt | 0.181 → 0.151 | 1.416 → 1.515 | 1.853 → 1.818 |

All 78 estimable full and blocked posteriors pass the declared numerical checks. Three quadrant partitions contain no test sites; every accepted row is still scored once under each deletion design. B-Tm, Mazama and YTT have worse geographic scores; Mazama's quadrant score also worsens materially. Full-fit upper-limit probability improves on average, while the range score becomes slightly worse. Those regressions remain in the tables. Geographic folds, quadrant deletions and survey tests rebuild geometry, basis, coefficients and model weights using training locations only.

[Case metrics and integrals](paired-results.csv) · [Paired site predictions](matched-sites.csv.gz) · [Paired geographic and quadrant predictions](matched-holdouts.csv.gz) · [Scores by observation kind](events.csv) · [Saved held-out predictive distributions](fold-marginals/) · [Matched regional benchmark](regional/comparison.csv)

## What changed in the geometry

The former growing short-distance covariance allowed adjacent distal readings to shape very large areas. Its shared basis also stopped at 96 modes, so densely sampled deposits could lose local contrast. The revised model gives broad and fine variation independent bases. The broad process remains fully coherent; the fine process has a fixed physical correlation length and separates coherent local contrasts from unresolved site variation. This reduces repeated shoulders and distant protrusions while preserving strong measured-site agreement. Removing the survey-dependent reach operator prevents the pattern of visited sites from acting as a thinning observation. Power-law and exponentially tempered power-law members remain available under one common prior.

The [visual and geological review](visual-review.md) follows each inner family through the intermediate panel and into the full deposit. It records curvature, spacing, eccentricity, orientation, topology and lobe transitions, and distinguishes supporting readings from conditional extrapolation. Each paired plate evaluates both posteriors directly on identical Cartesian windows. Actual case plates separately mark measured brackets with solid segments and conditional continuation with dashes. The three-scale probability plates show the uncertainty hidden by any single median line.

| Eruption | Identical-window comparison | Principal map and uncertainty |
| --- | --- | --- |
| Aso-4 | [Three scales](aso-4.png) | [Case](../../../eruptions/aso-4/README.md) |
| B-Tm | [Three scales](b-tm.png) | [Case](../../../eruptions/b-tm/README.md) |
| Ilopango TBJ | [Three scales](ilopango-tbj.png) | [Case](../../../eruptions/ilopango-tbj/README.md) |
| Mazama | [Three scales](mazama.png) | [Case](../../../eruptions/mazama/README.md) |
| Minoan | [Three scales](minoan.png) | [Case](../../../eruptions/minoan/README.md) |
| Pinatubo Layer C | [Three scales](pinatubo-1991-layer-c.png) | [Case](../../../eruptions/pinatubo-1991-layer-c/README.md) |
| Sakurajima Satsuma | [Three scales](sakurajima-satsuma.png) | [Case](../../../eruptions/sakurajima-satsuma/README.md) |
| Tambora | [Three scales](tambora.png) | [Case](../../../eruptions/tambora/README.md) |
| YTT | [Three scales](ytt.png) | [Case](../../../eruptions/ytt/README.md) |

[Geometry at two screening resolutions](geometry.csv.gz) includes topology, turning, eccentricity, axis changes, spacing and high angular frequencies. [Secondary-feature checks](secondary-features.csv.gz) identify enclosed readings and re-evaluate suspect holes or detached patches directly on finer local grids. The whole-Earth integration grid is only a screening mesh for these diagnostics; it does not determine the displayed contours. A small region with no enclosed point is not automatically erroneous: neighbouring readings can bracket it, while a loose upper limit cannot establish a thick patch. The visual review records these distinctions.

## Known deposits, surveys and structural sensitivity

The known-deposit comparison reruns the preserved v5 engine on exactly the same synthetic observations as v6, with matching input hashes and truths. It covers smooth deposits, several fans, turning fans, a broad umbrella, missing sectors, rotated survey spokes and an invisible-fan alternative. All 16 revised fits and their complete-draw integrations pass; one preceding integration fails its original convergence check. Fifteen of 18 tested truth totals lie inside the revised conditional 95% intervals. The two smooth-deposit totals and the visible-only alternative of one invisible-sector test lie just below their lower endpoints. The larger hidden-fan alternatives are covered. This small, deliberately diverse collection does not establish nominal 95% frequentist coverage.

The geometric recovery tradeoff is substantial. Mean contour intersection-over-union falls from 0.738 to 0.642; mean false area rises from about 46,360 to 123,526 km², while missed area falls from 35,275 to 22,157 km². These averages combine different thresholds, deposits and two spatial scrambles. Broader conditional tails improve recovery of turning and rotated-survey deposits but overextend several simple synthetic deposits. Mean absolute log error in median-map volume changes little, from 0.187 to 0.186. The published median is therefore not a validated boundary in an unsurveyed sector. No volume target, synthetic truth or eruption label enters the fitting objective.

[Paired recovery and input identities](paired-known-deposits.csv.gz) · [Current total intervals](known-volume-intervals.csv) · [Preceding total intervals](previous-known-volume-intervals.csv) · [All current integration refinements](known-volume-refinements.json) · [All preceding refinements](previous-known-volume-refinements.json) · [Rotated-survey radii](rotated-survey-radii.csv.gz)

The eight campaign and evidence-deletion tests retain their actual row identities, including Calbuco as an external eruption and interpreted non-detection constraints. They are additional transfer checks, not new observations inserted into the main fits. Both preceding and revised models are scored at the same withheld sites. [Survey predictions](survey-predictions.csv.gz) · [Training and withheld identities](survey-fits.json) · [Matched survey comparison](matched-surveys.csv.gz)

Negative log scores improve from 1.966 to 1.694 for the Aso-4 marine campaign, 3.510 to 3.290 for Pinatubo's withheld marine collection, and 1.899 to 1.861 for external Calbuco. The Mazama interpreted-bound deletion also improves. B-Tm, Ilopango and the separate Sihailongwan bound become slightly worse; the Pinatubo interpreted-bound score is nearly unchanged. These tests overlap some full-catalogue evidence and must not be counted as eight independent replications of the geographic result.

Changing the coherent fraction of residual fine-scale variation from one half to zero or one gives 27 passing fits. Median-map volumes vary much less than the local site agreement, with the largest relative spread in Minoan. These endpoint choices remain structural sensitivities, not per-eruption choices. [Sensitivity table](resolution-sensitivity.csv)

## Numerical evidence and scope

Exact readings are conditioned analytically. Bounded-event integration works in the constrained latent subspace, with analytic Gaussian control variates and checks against the original likelihood. Independent CDF checks include model-weight uncertainty. All 864 prior states and every attempted resolution are retained for every fitted partition. Posterior storage restores unconstrained directions as a full prior instead of deleting their uncertainty. Whole-deposit integration refines parameter populations separately from spatial quadrature, retaining the original acceptance thresholds. Median-map and quartile-map integrals remain distinct from total-draw intervals.

The [development archive](../../../archive/2026-11-26-directional-development/README.md) retains alternatives, unsuccessful numerical attempts and reasons for rejection. It includes covariance and source-gate changes, coherent-fraction ablations, reach ablations, pure-exponential extensions and learned-amplitude variants. Some recovered simple synthetic contours better but impaired real-data transfer or generated unsupported secondary patches. These comparisons informed method selection; the reported geographic and synthetic results are development evidence, not independent confirmation after a preregistered search. No single scalar score decided promotion.

[File identities and reproduction metadata](provenance.json) · [Final fit and challenge receipts](qualification-attempts.tar.gz) · [Numerical and map replay](../../../benchmarks/verify_directional_geography.py)

The reviewed method improves the observed-data tradeoff and the naturalness of the nine fitted contour families. It cannot infer unobserved lobes, distinguish erosion from original fallout at an isolated locality, or remove conditional uncertainty merely by producing a smoother-looking map. Supported local complexity and numerical qualifications remain visible in each case.
