# B-Tm · spherical-mixture global fallout estimate

[Eruption home](../../README.md) · [Regional GP/spline comparison](../README.md) · [Method and assumptions](../../../../docs/methods/probabilistic-global.md) · [All six eruptions](../../../../comparisons/spherical-mixture/README.md)

The baseline posterior median is **49.1 km³**, with a **30.7–342 km³ conditional 95% credible interval**. This is bulk fallout across all jointly fitted fans, including co-ignimbrite ash where present in the accepted data. It excludes ignimbrite deposits and is not dense-rock equivalent. Numerical diagnostics: **passed**.

Every posterior field is positive and has total volume equal (to the reported normalization tolerance) to its fitted volume parameter on the whole Earth. No closing pseudo-observations, map rectangle, contour cutoff or arbitrary distal radius sets this total. The previous contour screen remains a diagnostic of local support; its rejection does not veto this conditional global estimate.

![Global fallout field and uncertainty](field.png)

The overview expands around the displayed 0.1 cm contour, with a recorded finite-grid edge check. Its window never sets the volume. Fine coastlines are geographic context only. [Display bounds, posterior subset and coastline provenance](map-display.json).

![Local thickness and uncertainty around every accepted observation](local.png)

![Whole-Earth continuation and coastlines](world.png)

The local detail can cut contours at its edges; the whole-Earth companion shows their wider context. Longitude edges join in that companion. Colours saturate below 0.1 cm and above the displayed upper limit. Neither a smooth surface nor a closed contour establishes geological support.

## Source constraint

Every fitted field has its global thickness maximum at the supplied volcano. This is an explicit shape assumption, not an invented vent-thickness observation. Thickness decreases along every source ray, so this family cannot represent a secondary radial maximum. Real deposits can violate a point-source maximum, particularly with distributed sources or secondary deposition. The maximum relative change in any posterior component's integral on quadrupling angular resolution was **3.64e-09** (tolerance 10⁻⁶). [Derivation, rejected alternatives and geological limits](../../../../docs/methods/source-anchored-fans.md).

![Source-region thickness contours](source.png)

## What the interval means

Uncertainty includes the fitted field geometry, relative strengths, total volume and observation scatter parameters. The source-focused alternative also varies its tail shape; the bounded-envelope alternative has Gaussian tails. The latent volume excludes observation noise. It remains conditional on general modelling assumptions: a finite mixture of smooth, source-related, radially decreasing components, a proper volume/scale prior and a stated observation model. Several components form overlapping fallout fans; they are mathematical components, not independently measured eruption phases.

No finite dataset alone can exclude an arbitrarily large deposit in an unsampled region. The proper prior makes inference possible; it does not create geological evidence. Unrecorded erosion, reworking, preferential sampling and source misattribution are not automatically covered by this interval. If residual scatter is unresolved areal deposition variability rather than observation/preservation error, central-field volume can differ from area-mean physical mass; that structural correction is not automatically included. Bounds retain their documented likelihood meaning.

## Sensitivity to those assumptions

| Assumption | Median km³ | Conditional 95% interval km³ | Fitted-site log-RMSE | Numerical checks |
|---|---:|---:|---:|---|
| Source-focused fans · baseline | 49.1 | 30.7–342 | 0.813 | pass |
| Allow diffuse global tails | 49.5 | 30–396 | 0.815 | pass |
| Broader scale and shape priors | 54.4 | 31–1.23e+03 | 0.797 | pass |
| Shift volume prior 100-fold | 50 | 30.8–505 | 0.816 | pass |
| One fan | 48.5 | 30.9–66 | 0.829 | pass |
| Five fans | 50.4 | 30.8–362 | 0.783 | **not passed** |
| Gaussian observation errors | 38.7 | 25.6–232 | 0.667 | **not passed** |
| Diffuse fan and kernel weights | 52.5 | 32–191 | 0.770 | pass |

The range across the lower and upper interval endpoints is **25.6–1.23e+03 km³**. This is an assumption-sensitivity envelope, **not** a 95% interval, a probability-weighted model average, or an exhaustive bound. Every alternative remains visible, including those that differ substantially; none was selected because its volume matched a published target.

Fitted-site log-RMSE compares the posterior predictive median with the exact measurements used for fitting; bounds contribute their likelihood events instead. These training checks can expose a restrictive alternative, but are optimistic and **not validation or a model-selection rule**. Only the baseline receives the full matched geographic holdout analysis below. Alternatives have not been given probability weights or declared equally plausible.

![Volume sensitivity and mass partition](volume.png)

## Validation and numerical precision

Four buffered geographic holdouts: equal-region negative log score **1.492** (lower is better), log-thickness RMSE **1.490**, and heldout observation coverage **93.8%**. All four posterior fits passed numerical diagnostics: **False**. These predictions test transfer to withheld observations, not the unobserved whole-deposit volume.

Baseline chains: maximum rank-normalized split R-hat **1.0019**, minimum bulk/tail effective sample sizes **1560/698**, minimum energy BFMI **0.662**, divergences **0**. Diagnostics cover label-invariant global, observed-site and unsampled-field summaries. Passing them is evidence of numerical mixing, not proof of complete mode exploration. All shorter failed attempts are retained in the report. Monte Carlo standard errors of volume log-quantiles are recorded separately; rounded interval endpoints are estimates. [Inspect the chain traces and distributions](sampling.png).

Mass partition uses 1024 whole posterior fields and 4096 positive quadrature points per component. The largest fraction change on doubling that integration rule is **0.0080**. This numerical partition never changes the separately normalized total. Its local footprint is a sampling screen, not a confidence region.

## Reproduce and inspect

Use `tephra-isopachs global data/eruptions/B-Tm_cryptotephra.csv --geometry source-fans --output results/new-global-run`. Use `--resume` only for the same run; caches verify inputs, priors, numerical settings, source hashes, runtime versions and posterior checksums. The ordinary `compare` command runs the regional comparison followed by this global analysis.

- [Full report, priors, diagnostics, all attempts and provenance](report.json)
- [Baseline whole-field posterior samples](posterior.npz)
- [Volume samples for every assumption and chain](volume_draws.csv.gz)
- [Assumption summary](estimates.csv)
- [Geographic predictions and fold membership](heldout_predictions.csv)

Input SHA-256: `639871c9bc4e57109f8d0573e233dfb3113fa7c5be9de40d0c14a704e5d1bba0`. No accepted observations or source audits were changed for this calculation.

[Complete-contour support and geographic-deletion checks](complete-contour-audit.json).
