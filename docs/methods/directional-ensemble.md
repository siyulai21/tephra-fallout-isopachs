# Directional ensemble and median isopachs

This protocol describes the six underlying structures and the fixed-direction comparison. The principal directional reconstruction integrates the median map with uncertain fan direction. [Principal protocol](uncertain-azimuth.md)

The full directional run also evaluates continuation restricted to measured fan directions and the archive descriptions at their actual locations. The six-structure map provides the sensitivity to a hypothetical unseen fan. [Directional reach and archive evidence](archive-evidence.md)

The central reconstruction is a map of the **pointwise median latent fallout thickness**. Its published bulk volume is the integral of that same map over the Earth. Every case also has a distribution of volumes from complete simulated deposits, a table of how much lies in each compass direction and a comparison against the original directional fit. [Six-eruption results](../../comparisons/directional/fixed-ensemble/README.md) · [Underlying directional model](directional-continuation.md)

## Six plausible continuations

The ensemble crosses three nominal fan half-widths (22.5°, 45° and 90°) with two treatments of the largest gap in distal bearings. Three structures continue using the sampled bearings alone. The other three allow one additional uncertain fan centred in that gap. Its prior reach uses the 75th percentile of measured site radii, with the original broad multiplicative uncertainty. The site array determines the gap and scale anew within **each training fold**. Each isolated site supplies a fan of half its structure’s nominal width. All structures share the source-centred positive radial core, adaptive distal thinning, measurement triangles and transition rule. Lower and upper bounds enter the likelihood as events. No artificial thickness is assigned to a gap.

A plausible unsampled lobe matters especially when the observations lie on one side of the source. Its presence, absence or maximum possible size cannot generally be learned from these data. The six structures are a common, finite choice. Their widths, gap rule and weight regularization are modelling assumptions. The separately published single-structure sensitivity study varies tail shape, support footprint, background range and the unresolved proximal cap. These alternatives help assess uncertainty beyond the ensemble. [Directional assumptions and sensitivities](directional-continuation.md)

## Predictive weights and geographic validation

For each calibration eruption, each member is fitted to the same buffered geographic training folds. The held-out quantity is the posterior predictive density on log thickness at an exact observation, or probability of the recorded event at a bound. Observation and unresolved site scatter enter this calculation. Each case being reported is excluded in its entirety from weight fitting. A matching catalogue slug, input SHA-256 or vent coordinates within 0.01° triggers the exclusion. Longitude wraps at the dateline. Alternate files from a known source should retain their reviewed vent identity. An external eruption uses all available independent calibration cases.

With $p_{eifm}$ the held-out predictive density or event probability for member $m$, row $i$, fold $f$ and calibration eruption $e$, the weights maximize

$$
\sum_e {1\over F_e}\sum_f {1\over n_{ef}}\sum_i
    \log\!\left(\sum_{m=1}^6 w_m p_{eifm}\right)
       + {1\over6}\sum_{m=1}^6\log w_m,\qquad
   w_m>0,\quad \sum_m w_m=1.
$$

Each eruption has total weight one regardless of site count. The last term shrinks toward equal structure weights with the influence of one calibration eruption. A training fold enters calibration only if all six members are estimable and pass the common chain diagnostics. The calibration packet retains excluded folds and their reasons. A case with no usable calibration fold contributes zero weight to the fitting objective. If there are no independent usable eruptions, the model uses equal weights. Weight stability is assessed by 128 fixed-seed Bayesian bootstrap reweightings of whole calibration eruptions and by removing the regularizer. The reported volume interval conditions on the selected weights. Each report gives the effect of equal weights and of retaining only the three structures for each gap assumption. [Predictive stacking, Yao et al.](https://sites.stat.columbia.edu/gelman/research/published/stacking.pdf) · [Calibration fold identities, probabilities and exclusions](../../src/empirical_isopach/assets/directional-calibration.json)

For the target eruption, both the four buffered geographic folds and the four fixed compass quadrants are tested with the independently fitted weights. Every structure rebuilds its geometry from training rows. Target-fold predictions retain numerical warnings. Empty or infeasible deletions stay in the record. Predictive intervals invert the full weighted mixture of observation distributions. Holdout scores measure geographic thickness prediction. Catalogue reuse during method development and the small number of independent eruptions limit transfer claims.

## What a median isopach means

Let $F_m(t\mid x)$ be the conditional distribution of latent thickness at a location. The central thickness is the lower weighted median

$$
T_{50}(x)=\inf\{t:\ \sum_m w_m F_m(t\mid x)\geq 1/2\}.
$$

Above a contour of thickness $t$, the ensemble assigns more than half its mass to $T(x)>t$, with equality handled by the lower-quantile convention. This rule minimizes expected area misclassified as above or below the level under equal area costs. It preserves multiple lobes, islands and holes. A valid median surface can have curved isopachs even inside one triangle because a different draw can provide the median at each location.

The local evidence panel clips isopachs of the actual median display grid to triangles. Exact values at different vertices must bracket the level and at least 90% of weighted complete fields must cross that level in the triangle. The draw-by-draw test uses all three latent node values. Tables retain every decision and all displayed line vertices. The shared whole-contour screen independently tests bracketing, closure, holes, grid refinement and four usable geographic deletions. These display and support criteria leave the global estimator available when no complete band qualifies. [Complete-contour protocol](supported-isopachs.md)

## Two volume summaries

With thickness in cm and authalic area in km², the headline quantity is $10^{-5}\int T_{50}(x)\,dA$. It equals the volume represented by the central map, including its conditional continuation beyond sampled terrain. In a second calculation each draw selects one member for the entire planet and integrates one complete latent thickness field. The weighted 2.5th and 97.5th percentiles of those totals describe uncertainty in whole-deposit volume conditional on the structures, priors, accepted input and weights.

For example, consider three equal-area sectors with thickness 1 everywhere and an additional 9 units in exactly one sector, equally likely in each. Every complete field totals 12 volume units. The median map has thickness 1 in each sector and totals 3. A coherent-field interval can therefore exclude the integral of the median map. Pointwise thickness limits do not form a simultaneous contour band. Observational scatter enters neither volume.

## Numerical checks

Each structure contributes equally spaced midpoint indices from each of four independent chains. The nominal map uses 64 draws per chain per member. Within a member, every retained draw has mass $w_m/(4n_m)$. The algorithm sorts all latent values at each point and inverts the combined empirical CDF. The archive records the original draw indices, parameter draws, design and input hashes. Local, expanded and world maps use the same final field packet as the global median integral.

Two independently scrambled, nested Sobol designs cover the full equal-area disk out to the antipode. For $u,v\in[0,1]$, set $\theta=2\pi u$, $L=\log(1+2R/50)$ and $r=50[\exp(Lv)-1]$ km, where $R$ is the WGS84 authalic radius. The positive area weight is $2\pi Lr(r+50)$. The fixed 50 km scale places numerical points and leaves the physical integration domain unchanged. The five support classes crossed with eight compass sectors exhaust the Earth. Mean thickness on these nodes is independently checked against the weighted mean of **the same fields** integrated with analytic radial mass and triangle quadrature.

Each scramble uses 2,048 to 65,536 points. Refinement stops when the change across nested designs is below 1.5% of the total, the difference between scrambles is below 2.5% and the independent mean-integral discrepancy is below 1%. The first two criteria sum absolute discrepancies across all 40 cells. Two interleaved halves of every chain must each recover the central volume within 2.5%. At seven measured-range thickness levels, the two halves’ exceedance sets are compared by symmetric difference divided by their union with the central set. The threshold is 10% where the area-weighted union has effective sample size at least 64. Sparse levels are reported individually. Failing a draw check doubles the packet to 128 and then 256 draws per chain. The terminal discrepancy and all refinement attempts are retained. These are numerical tolerances, distinct from geological uncertainty.

The coherent-field integrals use all retained original draws for the six real eruptions. Each member has its own independent numerical partition check, volume-chain diagnostic, and original sampler attempts. Known-volume stress tests use the same map procedure and only the selected chain-balanced fields for the coherent-volume distribution. A passing numerical check says how the specified estimator behaved. Missing fans, lost deposits, unresolved source caps and post-depositional redistribution can move real volume outside its conditional interval. [Calibration and benchmark implementation](../../benchmarks/directional_ensemble.py)

## Reproduction

For a new input, the normal directional command fits the structures, uses the packaged out-of-eruption calibration, and writes the new map and reports. The full comparison command also runs the regional and spherical families.

```sh
tephra-isopachs global Minoan --geometry directional --output results/new-directional
tephra-isopachs compare Minoan --output results/new-complete
```

The following commands reproduce the catalogue study in scratch storage. Rebuild calibration only when publishing a new version of the common policy. The report generator reads the calibrated packet bundled under the model assets.

```sh
python -m benchmarks.directional_ensemble --work results/reproduction/cases
python -m benchmarks.directional_ensemble --stage calibrate --work results/reproduction/cases --output results/reproduction/calibration.json
python -m benchmarks.directional_ensemble --stage report --work results/reproduction/cases --output results/reproduction/reports
python -m benchmarks.verify_directional_ensemble
```

The packaged packet and the regenerated packet should agree in the recorded input hashes, member order and fold identities. Numerical predictive scores can depend on runtime and sampler version. The complete trial design uses seeds 90 and 91 with the seven choices printed by `python -m benchmarks.directional_ensemble --help`. [Trial driver](../../benchmarks/directional_ensemble.py)
