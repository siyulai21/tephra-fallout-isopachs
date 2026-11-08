# Start here: measurements to fallout estimates

[Project home](../README.md) · [Choose an eruption](../eruptions/README.md) · [Add an eruption](NEW_ERUPTION.md)

Field studies recover ash at scattered locations. This project fits thickness surfaces, draws equal-thickness contours (isopachs), integrates those surfaces, and checks predictions in withheld geographic regions. You can supply a new dataset in chat; its evidence and calculations are then published in this repository.

## Follow one complete example

1. Start at [Minoan](../eruptions/minoan/README.md). Its case page links every part of the calculation.
2. Read the [source audit](../data/audits/Minoan/README.md) and [accepted input](../data/eruptions/Minoan_tephra.csv): 92 exact thicknesses and nine lower bounds, combining all accepted Plinian and co-ignimbrite fallout fans.
3. Open the [regional report](../eruptions/minoan/results/README.md). GP ensemble and cubic spline fit the same evidence and are assessed on the same withheld regions.
4. Open the [support audit](../eruptions/minoan/results/support/README.md). It identifies sampling gaps, local footprints and rejected contours.
5. Open the [global report](../eruptions/minoan/results/global/README.md). A separate mass-normalized model estimates whole-deposit fallout under explicit assumptions, with posterior uncertainty and prior sensitivity.

## Understand the three reports

| Report | What it can tell you | What it does not establish |
|---|---|---|
| Regional GP/spline comparison | Near-data reconstruction, held-out prediction and volume over the common observation hull | Complete-deposit volume or uniformly supported interpolation inside gaps |
| Local support audit | Where sampling geometry supports interpolation and which contours survive declared checks | That a closed contour validates its interior, or that partial volume is a certified lower bound |
| Conditional global posterior | Whole-Earth fallout implied by a fitted field and its priors | That unobserved mass is identified by measurements alone |

The GP ensemble and cubic spline are different statistical methods sharing data and evaluation. The global companion has a different volume domain and uncertainty contract. A low regional prediction error does not by itself resolve extrapolated global mass. [Method catalogue](../methods/README.md).

All volumes describe **bulk fallout**. Dense-rock equivalent needs independent density information; total eruption volume may also include ignimbrite deposits that are outside this calculation.

## Read evidence before a headline number

Missing locations are not zeros. A documented detection limit can supply an upper bound; an incomplete deposit can sometimes supply a qualified lower bound. Source selection and geological interpretation remain explicit even though statistical fitting is automated.

Inspect held-out scores, coverage, numerical warnings and assumption sensitivity. Regional conditional intervals omit fitted-parameter and selection uncertainty. Global credible intervals are conditional on their model and priors; the span across sensitivity refits is not another 95% interval. A failed support screen does not veto the global calculation, but highlights the missing evidence.

For examples of why the audits matter, read [Aso-4's corrected coordinates and unresolved global tail](../comparisons/aso-4-audit/README.md) or [B-Tm's source-contour correction and remaining thickness qualifications](../eruptions/b-tm/README.md).

## Your next step

[Upload measurements](NEW_ERUPTION.md) · [Run locally](QUICKSTART.md) · [Read output fields](OUTPUTS.md) · [Compare validation evidence](VALIDATION.md) · [Find a file](REPOSITORY_MAP.md) · [Agent task routes](../AGENTS.md)
