# Sensitivities for unseen fan volume and displaced thickness maxima

The two alternatives preserve the existing target inputs and the published principal fields. Each keeps one coherent global field per possible deposit. Their conditional intervals summarize complete-deposit integrals. Matched holdout comparisons use the same posterior subset for the candidate and its control. A lower score estimates better prediction at withheld sampled sites. [Cross-eruption comparison](../../comparisons/nested-isopachs/README.md)

## Directional gap amount

Let `h(x)` be the existing directional exposure, zero at the training sites and inside the retained interpolation footprint. Multiply a complete uncertain-bearing field by `F^h(x)`, using a common `F` for that entire draw. The declared factors ¼, 1 and 4 have probabilities ¼, ½ and ¼. This adds a distinct uncertainty about how much fallout entered the unsampled bearings. It is symmetric on the log scale. It can change the expected volume. The observations leave these odds unidentified because every factor equals one at them.

Two independent scrambled rules integrate each of the three fields on the full authalic disk. The report gives the interval for each factor, the mixture interval, the agreement under doubling and between scrambles, and recovery of the published control. Each training fold rebuilds its bearing exposure, reuses the verified training-only posterior and scores the same withheld measurements and bounds. Maps compare two pointwise medians on the same survey frame. [Implementation](../../src/empirical_isopach/models/directional/gap_amount.py)

## Spherical source displacement

For displaced components, the fitted offset vector is the stereographic centre displacement in units of the component scale. Reweight the existing prior by `1/(1 + sum(w_j |offset_j|²)/4)`. The tilt is bounded, positive and common to every eruption. It reduces the prior probability of large displacement relative to the width of components that carry substantial volume. An off-source peak remains possible. Source-focused components have no displacement parameter and receive a factor of one.

The same prior ratio gives posterior importance weights on the original complete fields. The bearing factors and their complete-Earth volume integrals are unchanged in each draw. Effective draw counts, between-chain volume medians and training-fold diagnostics accompany the new quantiles. A large shift or weight concentration warrants a direct refit before promotion. The paired geographic check uses the signed support-packet subset from each training fold. Its control score may differ slightly from the full-posterior headline score. [Implementation](../../src/empirical_isopach/models/mass_mixture/displacement.py)
