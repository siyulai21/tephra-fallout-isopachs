# Aso-4 · global fallout estimate

[Eruption home](../../README.md) · [Regional GP/spline comparison](../README.md) · [Method and assumptions](../../../../docs/methods/probabilistic-global.md) · [All six eruptions](../../../../comparisons/global/README.md)

**Aso-4 source qualification:** this is the reviewed coordinate-and-bound scenario, including the unresolved Biwa source conflict and an erosion interpretation. The high global total remains questioned. A tighter fan-position alternative predicts heldout regions better but does not independently validate global mass. Read the [source audit, marine-setting tests and all targeted alternatives](../../../../comparisons/aso-4-audit/README.md).

The baseline posterior median is **1.47e+03 km³**, with a **501–4.42e+04 km³ conditional 95% credible interval**. This is bulk fallout across all jointly fitted fans, including co-ignimbrite ash where present in the accepted data. It excludes ignimbrite deposits and is not dense-rock equivalent. Numerical diagnostics: **passed**.

Every posterior field is positive and has total volume exactly equal to its fitted volume parameter on the whole Earth. No closing pseudo-observations, map rectangle, contour cutoff or arbitrary distal radius sets this total. The previous contour screen remains a diagnostic of local support; its rejection does not veto this conditional global estimate.

![Global fallout field and uncertainty](field.png)

## What the interval means

Uncertainty includes fan positions, sizes, relative masses, tail shape, total volume and observation scatter parameters. The latent volume excludes observation noise. It remains conditional on general modelling assumptions: a finite mixture of smooth, source-related, radially decreasing components, a proper volume/scale prior and a stated observation model. Several components can form elongated, separated or overlapping fans; they are mathematical components, not independently measured eruption phases.

No finite dataset alone can exclude an arbitrarily large deposit in an unsampled region. The proper prior makes inference possible; it does not create geological evidence. Unrecorded erosion, reworking, preferential sampling and source misattribution are not automatically covered by this interval. If residual scatter is unresolved areal deposition variability rather than observation/preservation error, central-field volume can differ from area-mean physical mass; that structural correction is not automatically included. Bounds retain their documented likelihood meaning.

## Sensitivity to those assumptions

| Assumption | Median km³ | Conditional 95% interval km³ | Fitted-site log-RMSE | Numerical checks |
|---|---:|---:|---:|---|
| Three fans · baseline | 1.47e+03 | 501–4.42e+04 | 0.594 | pass |
| Allow diffuse global tails | 1.62e+03 | 487–3.73e+04 | 0.591 | pass |
| Broader scales and fan offsets | 3.82e+03 | 707–2.16e+05 | 0.567 | pass |
| Shift volume prior 100-fold | 1.71e+03 | 510–6.72e+04 | 0.591 | pass |
| One fan | 3.1e+05 | 4.41e+03–1.41e+08 | 0.756 | pass |
| Five fans | 1.78e+03 | 520–4.07e+04 | 0.583 | pass |
| Gaussian observation errors | 1.47e+03 | 496–4.01e+04 | 0.594 | pass |
| Diffuse fan and kernel weights | 1.1e+03 | 465–6.51e+03 | 0.593 | pass |

The range across the lower and upper interval endpoints is **465–1.41e+08 km³**. This is an assumption-sensitivity envelope, **not** a 95% interval, a probability-weighted model average, or an exhaustive bound. Every alternative remains visible, including those that differ substantially; none was selected because its volume matched a published target.

Fitted-site log-RMSE compares the posterior predictive median with the exact measurements used for fitting; bounds contribute their likelihood events instead. These training checks can expose a restrictive alternative, but are optimistic and **not validation or a model-selection rule**. Only the baseline receives the full matched geographic holdout analysis below. Alternatives have not been given probability weights or declared equally plausible.

![Volume sensitivity and mass partition](volume.png)

## Validation and numerical precision

Four buffered geographic holdouts: equal-region negative log score **1.632** (lower is better), log-thickness RMSE **1.254**, and heldout observation coverage **95.2%**. All four posterior fits passed numerical diagnostics: **True**. These predictions test transfer to withheld observations, not the unobserved whole-deposit volume.

Baseline chains: maximum rank-normalized split R-hat **1.0024**, minimum bulk/tail effective sample sizes **1401/1553**, minimum energy BFMI **0.489**, divergences **0**. Diagnostics cover label-invariant global, observed-site and unsampled-field summaries. Passing them is evidence of numerical mixing, not proof of complete mode exploration. All shorter failed attempts are retained in the report. Monte Carlo standard errors of volume log-quantiles are recorded separately; rounded interval endpoints are estimates. [Inspect the chain traces and distributions](sampling.png).

Mass partition uses 1024 whole posterior fields and 4096 equal-mass quadrature points per fan. The largest fraction change on doubling that integration rule is **0.0052**. This numerical partition never changes the analytically normalized total. Its local footprint is a sampling screen, not a confidence region.

## Reproduce and inspect

Use `tephra-isopachs global data/eruptions/Aso-4_reviewed.csv --output results/new-global-run`. Use `--resume` only for the same run; caches verify inputs, priors, numerical settings, source hashes, runtime versions and posterior checksums. The ordinary `compare` command runs the regional comparison followed by this global analysis.

- [Full report, priors, diagnostics, all attempts and provenance](report.json)
- [Baseline whole-field posterior samples](posterior.npz)
- [Volume samples for every assumption and chain](volume_draws.csv.gz)
- [Assumption summary](estimates.csv)
- [Geographic predictions and fold membership](heldout_predictions.csv)

Input SHA-256: `20ff78ad97ea6c3c427d2866b4134a93cc39081897ea4afa8ec50ad1b6d864b9`. No accepted observations or source audits were changed for this calculation.
