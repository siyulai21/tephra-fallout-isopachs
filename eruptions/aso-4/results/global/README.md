# Aso-4 · global fallout estimate

**Source-geometry review (version 0.11):** This high total remains a questioned conditional estimate. Source-focused ellipses predict withheld observations better but raise the median to about 3,418 km³. Two lower-volume envelope candidates give about 472 and 369 km³, but predict withheld data worse and are not adopted. [Full alternatives, maps, provenance and limitations](../../../../comparisons/source-revision/README.md). The numerical results below are the preserved preceding calculation, not a new fit.

[Eruption home](../../README.md) · [Regional GP/spline comparison](../README.md) · [Method and assumptions](../../../../docs/methods/probabilistic-global.md) · [All six eruptions](../../../../comparisons/global/README.md)

The baseline posterior median is **1.84e+03 km³**, with a **507–5e+04 km³ conditional 95% credible interval**. This is bulk fallout across all jointly fitted fans, including co-ignimbrite ash where present in the accepted data. It excludes ignimbrite deposits and is not dense-rock equivalent. Numerical diagnostics: **passed**.

Every posterior field is positive and has total volume exactly equal to its fitted volume parameter on the whole Earth. No closing pseudo-observations, map rectangle, contour cutoff or arbitrary distal radius sets this total. The previous contour screen remains a diagnostic of local support; its rejection does not veto this conditional global estimate.

![Global fallout field and uncertainty](field.png)

## What the interval means

Uncertainty includes fan positions, sizes, relative masses, tail shape, total volume and observation scatter parameters. The latent volume excludes observation noise. It remains conditional on general modelling assumptions: a finite mixture of smooth, source-related, radially decreasing components, a proper volume/scale prior and a stated observation model. Several components can form elongated, separated or overlapping fans; they are mathematical components, not independently measured eruption phases.

No finite dataset alone can exclude an arbitrarily large deposit in an unsampled region. The proper prior makes inference possible; it does not create geological evidence. Unrecorded erosion, reworking, preferential sampling and source misattribution are not automatically covered by this interval. If residual scatter is unresolved areal deposition variability rather than observation/preservation error, central-field volume can differ from area-mean physical mass; that structural correction is not automatically included. Bounds retain their documented likelihood meaning.

## Sensitivity to those assumptions

| Assumption | Median km³ | Conditional 95% interval km³ | Fitted-site log-RMSE | Sampling diagnostics |
|---|---:|---:|---:|---|
| Three fans · baseline | 1.84e+03 | 507–5e+04 | 0.585 | pass |
| Allow diffuse global tails | 1.94e+03 | 502–5.05e+04 | 0.586 | pass |
| Broader scales and fan offsets | 4.27e+03 | 747–2.46e+05 | 0.572 | pass |
| Shift volume prior 100-fold | 2.14e+03 | 524–7.57e+04 | 0.586 | pass |
| One fan | 7.04e+05 | 6.65e+03–3.7e+08 | 0.743 | pass |
| Five fans | 1.99e+03 | 519–5.34e+04 | 0.582 | pass |
| Gaussian observation errors | 1.79e+03 | 501–6.12e+04 | 0.585 | pass |
| Diffuse fan and kernel weights | 1.15e+03 | 482–7.15e+03 | 0.589 | pass |

The range across the lower and upper interval endpoints is **482–3.7e+08 km³**. This is an assumption-sensitivity envelope, **not** a 95% interval, a probability-weighted model average, or an exhaustive bound. Every alternative remains visible, including those that differ substantially; none was selected because its volume matched a published target.

Fitted-site log-RMSE compares the posterior predictive median with the exact measurements used for fitting; bounds contribute their likelihood events instead. These training checks can expose a restrictive alternative, but are optimistic and **not validation or a model-selection rule**. Only the baseline receives the full matched geographic holdout analysis below. Alternatives have not been given probability weights or declared equally plausible.

![Volume sensitivity and mass partition](volume.png)

## Validation and numerical precision

Four buffered geographic holdouts: equal-region negative log score **1.644** (lower is better), log-thickness RMSE **1.248**, and heldout observation coverage **94.3%**. All four posterior fits passed numerical diagnostics: **True**. These predictions test transfer to withheld observations, not the unobserved whole-deposit volume.

Baseline chains: maximum rank-normalized split R-hat **1.0083**, minimum bulk/tail effective sample sizes **586/804**, minimum energy BFMI **0.520**, divergences **0**. Diagnostics cover label-invariant global, observed-site and unsampled-field summaries. Passing them is evidence of numerical mixing, not proof of complete mode exploration. All shorter failed attempts are retained in the report. Monte Carlo standard errors of volume log-quantiles are recorded separately; rounded interval endpoints are estimates. [Inspect the chain traces and distributions](sampling.png).

Mass partition uses 1024 whole posterior fields and 4096 equal-mass quadrature points per fan. The largest fraction change on doubling that integration rule is **0.0057**. This numerical partition never changes the analytically normalized total. Its local footprint is a sampling screen, not a confidence region.

## Reproduce and inspect

Use `tephra-isopachs global data/eruptions/Aso-4_tephra.csv --output results/new-global-run`. Use `--resume` only for the same run; caches verify inputs, priors, numerical settings, source hashes, runtime versions and posterior checksums. The ordinary `compare` command runs the regional comparison followed by this global analysis.

- [Full report, priors, diagnostics, all attempts and provenance](report.json)
- [Baseline whole-field posterior samples](posterior.npz)
- [Volume samples for every assumption and chain](volume_draws.csv.gz)
- [Assumption summary](estimates.csv)
- [Geographic predictions and fold membership](heldout_predictions.csv)

Input SHA-256: `802f22266d905b8d10c0535a7767e7bd83fb7f797692ea869095a3fa410360d8`. No accepted observations or source audits were changed for this calculation.
