# Compare the four methods

This figure asks how well each method predicts ash thickness in a withheld region. Lower scores are better. Percentages give the share of exact observations within the 95% predictive interval. A cross marks a global fit with a numerical warning or incomplete scoring. The directional series uses the current six-structure ensemble.

![Geographic prediction scores and exact-site coverage for the four methods across six eruptions](method-scores.png)

## Read the evidence by question

| Question | Study |
|---|---|
| How sensitive are directional isopachs and whole-Earth volume to the width and presence of fallout fans? | [Directional ensemble](directional-ensemble/README.md) — mapped volumes, complete-field intervals, withheld sectors and known-deposit tests |
| How much should a directional field extend into an unsampled sector? | [Six-eruption geographic and quadrant tests](bearing-continuation/README.md) and the [B-Tm archive check](../eruptions/b-tm/results/directional/bearing/README.md) |
| How much does the choice of regional fit matter inside the same observation hull? | [GP and cubic spline](regional/README.md) — matched geographic folds, hull integrals and synthetic trials |
| How much do global fan geometry and priors affect the spherical-mixture result? | [Spherical mixture](spherical-mixture/README.md) — separate posterior, prior refits and numerical warnings |
| Which contours or partial volumes have local observational support? | [Support audit](support/README.md) — sampling gaps, rejected bands and geographic deletions |

The directional ensemble predicted held-out thickness better than its single-structure reference in **five of six** eruptions and better than equal weights in **four of six**. Its conditional volume intervals contained the known total in **7 of 14** simulated trials and missed the larger hidden-fan total in both trials that allowed it. The real eruptions have no known whole-deposit volume against which to rank the methods. [Trial results and limitations](directional-ensemble/README.md#known-deposits-and-unseen-fans)

## Studies that explain specific results

- The [single-structure directional study](directional/README.md) preserves the original comparison with the spherical mixture, 13 assumption settings, independent numerical replications and an unmeasured source-cap experiment.
- The [Aso-4 source audit and influence tests](aso-4-audit/README.md) explain the corrected cruise positions, thickness bounds and questioned upper tail.
- The [B-Tm and Aso-4 source-geometry study](source-revision/README.md) explains B-Tm’s source-focused spherical fit and tests changes to its lake evidence.

For an individual eruption’s maps and estimates, open its [case home](../eruptions/README.md).
