# Minoan · global fallout estimate

[Eruption home](../../README.md) · [Regional GP/spline comparison](../README.md) · [Method and assumptions](../../../../docs/methods/probabilistic-global.md) · [All six eruptions](../../../../comparisons/global/README.md)

The baseline posterior median is **46 km³**, with a **28.1–2.53e+03 km³ conditional 95% credible interval**. This is bulk fallout across all jointly fitted fans, including co-ignimbrite ash where present in the accepted data. It excludes ignimbrite deposits and is not dense-rock equivalent. Numerical diagnostics: **passed**.

Every posterior field is positive and has total volume exactly equal to its fitted volume parameter on the whole Earth. No closing pseudo-observations, map rectangle, contour cutoff or arbitrary distal radius sets this total. The previous contour screen remains a diagnostic of local support; its rejection does not veto this conditional global estimate.

![Global fallout field and uncertainty](field.png)

## What the interval means

Uncertainty includes fan positions, sizes, relative masses, tail shape, total volume and observation scatter parameters. The latent volume excludes observation noise. It remains conditional on general modelling assumptions: a finite mixture of smooth, source-related, radially decreasing components, a proper volume/scale prior and a stated observation model. Several components can form elongated, separated or overlapping fans; they are mathematical components, not independently measured eruption phases.

No finite dataset alone can exclude an arbitrarily large deposit in an unsampled region. The proper prior makes inference possible; it does not create geological evidence. Unrecorded erosion, reworking, preferential sampling and source misattribution are not automatically covered by this interval. If residual scatter is unresolved areal deposition variability rather than observation/preservation error, central-field volume can differ from area-mean physical mass; that structural correction is not automatically included. Bounds retain their documented likelihood meaning.

## Sensitivity to those assumptions

| Assumption | Median km³ | Conditional 95% interval km³ | Fitted-site log-RMSE | Sampling diagnostics |
|---|---:|---:|---:|---|
| Three fans · baseline | 46 | 28.1–2.53e+03 | 0.973 | pass |
| Allow diffuse global tails | 63.5 | 29.2–4.04e+03 | 0.969 | pass |
| Broader scales and fan offsets | 65.4 | 29.8–2.12e+04 | 0.973 | pass |
| Shift volume prior 100-fold | 50.7 | 28.5–5.01e+03 | 0.972 | pass |
| One fan | 35.7 | 26.4–51.7 | 0.984 | pass |
| Five fans | 55 | 28.5–3.54e+03 | 0.971 | pass |
| Gaussian observation errors | 46.6 | 28–3.09e+03 | 0.970 | pass |
| Diffuse fan and kernel weights | 66.7 | 31.6–547 | 0.980 | pass |

The range across the lower and upper interval endpoints is **26.4–2.12e+04 km³**. This is an assumption-sensitivity envelope, **not** a 95% interval, a probability-weighted model average, or an exhaustive bound. Every alternative remains visible, including those that differ substantially; none was selected because its volume matched a published target.

Fitted-site log-RMSE compares the posterior predictive median with the exact measurements used for fitting; bounds contribute their likelihood events instead. These training checks can expose a restrictive alternative, but are optimistic and **not validation or a model-selection rule**. Only the baseline receives the full matched geographic holdout analysis below. Alternatives have not been given probability weights or declared equally plausible.

![Volume sensitivity and mass partition](volume.png)

## Validation and numerical precision

Four buffered geographic holdouts: equal-region negative log score **1.570** (lower is better), log-thickness RMSE **1.224**, and heldout observation coverage **94.6%**. All four posterior fits passed numerical diagnostics: **True**. These predictions test transfer to withheld observations, not the unobserved whole-deposit volume.

Baseline chains: maximum rank-normalized split R-hat **1.0048**, minimum bulk/tail effective sample sizes **869/1287**, minimum energy BFMI **0.601**, divergences **0**. Diagnostics cover label-invariant global, observed-site and unsampled-field summaries. Passing them is evidence of numerical mixing, not proof of complete mode exploration. All shorter failed attempts are retained in the report. Monte Carlo standard errors of volume log-quantiles are recorded separately; rounded interval endpoints are estimates. [Inspect the chain traces and distributions](sampling.png).

Mass partition uses 1024 whole posterior fields and 4096 equal-mass quadrature points per fan. The largest fraction change on doubling that integration rule is **0.0083**. This numerical partition never changes the analytically normalized total. Its local footprint is a sampling screen, not a confidence region.

## Reproduce and inspect

Use `tephra-isopachs global data/eruptions/Minoan_tephra.csv --output results/new-global-run`. Use `--resume` only for the same run; caches verify inputs, priors, numerical settings, source hashes, runtime versions and posterior checksums. The ordinary `compare` command runs the regional comparison followed by this global analysis.

- [Full report, priors, diagnostics, all attempts and provenance](report.json)
- [Baseline whole-field posterior samples](posterior.npz)
- [Volume samples for every assumption and chain](volume_draws.csv.gz)
- [Assumption summary](estimates.csv)
- [Geographic predictions and fold membership](heldout_predictions.csv)

Input SHA-256: `fe2a2e70cecbbfee8ea6a410ba282449e020c21598f6ecf7f9b6e51f15d7d1a1`. No accepted observations or source audits were changed for this calculation.
