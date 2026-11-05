# Start here

[Project home](../README.md) · [Reports](../eruptions/README.md) · [Methods](METHODS.md)

Field studies recover ash at scattered locations. This project reconstructs a thickness surface, draws equal-thickness contours (isopachs), integrates the surface, and tests predictions in withheld geographic regions.

## Follow one eruption

1. Read the [Minoan source audit](../data/audits/Minoan/README.md).
2. The [accepted data](../data/eruptions/Minoan_tephra.csv) contain 92 exact thicknesses and nine lower bounds, combining all accepted Plinian and co-ignimbrite fallout fans.
3. The [current comparison](../eruptions/minoan/results/README.md) fits a GP ensemble and a cubic spline to those same observations.
4. Compare their maps, regional volumes, withheld predictions and uncertainty checks. Neither is automatically selected as the winner.
5. Read the [global fallout analysis](../eruptions/minoan/results/global/README.md) for a joint estimate across all fans, with explicit uncertainty and sensitivity to general modelling assumptions.

## What happens in a calculation?

```text
Source measurements and geological notes
    → audit units, locations, deposit meaning and exclusions
Accepted exact thicknesses and documented bounds
    → fit and tune each method inside geographic training folds
GP ensemble                     Cubic spline
    → identical withheld regions and regional integration footprint
Paired maps, volumes, diagnostics and a reproducible report
```

A missing location is not a zero. An incomplete core can provide a lower bound; a documented detection limit can provide an upper limit. Predictions and numerical integration points never become additional observations.

## What is being compared?

The GP ensemble combines spatial covariance models. The spline fits cubic basis coefficients with automatically selected knots and smoothing. They are separate methods sharing observation semantics, evaluation and reporting. [Equations and differences](METHODS.md).

Both volumes are integrated over the **convex hull of the accepted locations**. This footprint can contain unsampled gaps and omit proximal deposits. The result is regional bulk fallout, not complete eruption volume. Bulk volume includes pore space; dense-rock equivalent needs independent density information.

## What deserves caution?

Start with the input audit, then inspect withheld predictive scores, errors and coverage. A smooth-looking map alone does not establish accuracy. Conditional intervals omit parameter/selection uncertainty and can be too narrow, especially when a spline strongly follows its fitted trend. Geographic-deletion ranges show a separate sensitivity, not a confidence interval.

Numerical convergence checks arithmetic, not geological correctness. The new global model estimates a positive field with mass normalized on the whole Earth, so no chosen map boundary controls the total. Its conditional credible interval and assumption-sensitivity envelope have different meanings. A failed local contour screen does not veto that global calculation; it highlights the missing evidence. Historical global estimates remain available but do not rank current methods.

## Navigate the repository

- [Eruptions](../eruptions/README.md): evidence, current reports and history for each case.
- [Methods](../methods/README.md): one GP ensemble and one automatic cubic spline, with room for future families.
- [Comparisons](../comparisons/README.md): shared regional evaluation across real and simulated deposits.
- [Add an eruption](NEW_ERUPTION.md): upload in chat, audit, calculate and publish on GitHub.
- [Archive](../archive/README.md): earlier analyses, clearly separated from current results.

[Quick start](QUICKSTART.md) · [Output guide](OUTPUTS.md) · [Data guide](../data/README.md) · [Repository map](REPOSITORY_MAP.md)
