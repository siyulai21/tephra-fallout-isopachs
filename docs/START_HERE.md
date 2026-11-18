# Start here: measurements to fallout estimates

[Project home](../README.md) · [Choose an eruption](../eruptions/README.md) · [Upload a dataset](NEW_ERUPTION.md)

Field studies find ash at scattered locations. This project reconstructs thickness surfaces, draws equal-thickness contours (isopachs), integrates each surface and tests predictions in withheld regions. Upload measurements and sources in chat; the assistant audits them and publishes the data, maps and reports here.

## Follow Minoan from evidence to results

1. Open the [Minoan case home](../eruptions/minoan/README.md), then its [source audit](../data/audits/Minoan/README.md) and [accepted table](../data/eruptions/Minoan_tephra.csv). The calculation combines all accepted Plinian and co-ignimbrite fallout: 92 exact thicknesses and nine lower bounds.
2. Read the [regional comparison](../eruptions/minoan/results/README.md). The GP ensemble and cubic spline independently fit the same observations, share their regional integration hull and use matched geographic holdouts.
3. Inspect the [local support audit](../eruptions/minoan/results/support/README.md). It preserves unsampled holes and shows why a smooth or closed contour is not automatically supported evidence.
4. Read **both** whole-deposit reconstructions: [directional continuation](../eruptions/minoan/results/directional/README.md) and [spherical mixture](../eruptions/minoan/results/spherical-mixture/README.md). Each starts from the accepted observations and makes different explicit assumptions across unmeasured ground.
5. Compare [cross-eruption evidence](../comparisons/README.md), including held-out prediction, known-volume failures, numerical warnings and prior sensitivity.

## Four families, not a chain of models

| Question | Methods | Meaning of the volume |
|---|---|---|
| How does the deposit fit near the observations? | GP ensemble and automatic cubic spline | Bulk fallout within the same observation hull |
| What whole-deposit fallout follows under stated continuation assumptions? | Directional continuation and spherical mixture | Conditional whole-Earth bulk fallout |
| Where is there local evidence? | Shared support audit, applied to every family | Diagnosed footprint, partial volumes and rejected contours |

The directional method is the latest **addition**, not a GP post-processing step, a new cubic spline or a replacement for the spherical mixture. Each family produces its own field and volume. No family is automatically preferred. [How the methods relate](../methods/README.md).

All volumes describe **bulk fallout**. Dense-rock equivalent needs independent density information. Ignimbrite deposits themselves are outside this calculation, even when co-ignimbrite airborne fallout is included.

## Read evidence before a headline number

Missing ground is not zero fallout. A documented detection limit can supply an upper bound; an incomplete deposit can sometimes supply a qualified lower bound. Geological source selection remains explicit even when statistical fitting is automatic.

Read conditional intervals alongside held-out coverage, failed runs and sensitivity. Regional intervals omit some fitted-parameter and selection uncertainty. Global intervals depend on model and prior assumptions; their endpoints do not bound every possible hidden deposit. The span across assumption refits is not another 95% interval. A failed contour screen diagnoses missing support without suppressing a conditional global estimate.

[Upload measurements](NEW_ERUPTION.md) · [Run locally](QUICKSTART.md) · [Read output fields](OUTPUTS.md) · [Find a file](REPOSITORY_MAP.md) · [Agent task routes](../AGENTS.md)
