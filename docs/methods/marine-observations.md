# Marine cores as observations of fallout

[Aso-4 audit](../../data/audits/Aso-4/README.md) · [Experiments](../../comparisons/aso-4-audit/README.md) · [Global field model](probabilistic-global.md)

A correlated ash layer can reliably identify an eruption without preserving its original atmospheric loading at that location. Treat source identification, thickness measurement and depositional preservation as distinct pieces of evidence. The Aso-4 review records documented bounds and coordinate corrections, retains uncertain records, and tests all explicitly marine records as a group.

## A mass-conserving observation operator

Let `a(u)` be atmospheric fallout mass per unit area. A marine core's preserved ash mass loading is more generally

`m(x) = p(x) ∫ K(x,u) a(u) dA(u) + r(x)`.

Here `K` is a nonnegative transport kernel, `p` is the preserved fraction and `r` represents material from an additional process or event. Pure redistribution conserves total mass if `∫ K(x,u) dA(x) = 1` for each starting location `u`, `p = 1`, and `r = 0`. This is a column-normalization requirement; a spatial smoothing rule normalized only over origins does not generally guarantee conservation on an irregular domain. Local core thickness can rise or fall even when the total atmospheric mass is unchanged.

This equation is a proposed framework, **not a transport model fitted in this release**. Converting `m(x)` to observed thickness additionally needs deposit bulk density, porosity and compaction assumptions. Conserving particle mass is not the same as conserving measured bulk thickness-volume when those quantities vary. A flexible fallout field plus an unknown flexible transport kernel is not identified by sparse thickness observations alone. Assigning a convenient current path, common marine inflation factor or arbitrary extra variance would replace missing evidence with assumptions.

## What is implemented now

1. Retain setting and primary-source evidence independently of the thickness residual. Do not classify a record as reworked merely because a model predicts it poorly.
2. Use documented upper/lower observations through their likelihood events. Do not convert dispersed shards, patches or missing horizons to a numerical thickness or a known detection limit.
3. Report the reference whole-deposit posterior with its geometry/prior sensitivities and explicit omission of unidentified transport uncertainty.
4. Refit after removing every source-labelled submarine record. Compare both volume and predictions at the omitted records. This tests dependence on the marine subset without supplying a correction factor.
5. Keep the standard buffered geographic validation separate. Setting removal confounds observation process with spatial coverage and is not a causal transport test or a replacement predictive ranking.

The current robust observation mixture can absorb isolated discrepancies. It does not model coherent transport of several cores, preferential sampling, or preservation bias. Its conditional 95% volume interval is therefore not advertised as a 95% interval covering all marine-process uncertainty. The separate geometry and setting alternatives have no assigned probabilities; their extrema are not a calibrated confidence interval.

## Evidence needed for a fitted extension

Useful information includes full shard-concentration profiles and integrated ash mass, verified layer contacts, grain size and density, sedimentary structures, chemical/stratigraphic correlation, core recovery and compaction, independently characterized non-detections, and spatially paired marine/terrestrial or closely spaced core observations. These can constrain an observation operator or a hierarchical setting model without inventing eruption-specific wind or umbrella parameters. Whole-profile ash mass can be preferable to the thickness of a visible band when bioturbation spreads the same ash vertically.

Any future model must conserve the intended quantity, include simulated recoverability and misspecification tests, fit observation-process parameters using training data only, and report prior sensitivity. Better local prediction alone still cannot validate unobserved global mass.

The geological basis is [Machida & Arai (1988)](https://doi.org/10.4116/jaqua.26.3_227), [Ikehara (2015)](https://doi.org/10.18940/kazan.60.2_181), and his [English review](https://doi.org/10.1186/s40645-015-0068-z). Their evidence motivates explicit observation-process uncertainty; it supplies no universal numerical marine correction.
