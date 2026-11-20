# Directional extrapolation in unsampled bearings

Both candidates predict the same buffered geographic and whole-quadrant holdouts from six eruptions. The measured-fan candidate uses three observation-derived fan widths and four cross-bearing range ratios. The comparator uses three widths with and without a possible fan in the largest distal gap. Both receive weights trained on the other eruptions using the same fold and eruption balance. Smaller scores indicate better prediction.

| Eruption | Geographic, measured fans / comparator | Quadrants, measured fans / comparator |
|---|---:|---:|
| mazama | 1.454 / 1.420 | 2.167 / 1.961 |
| tambora | 2.086 / 1.941 | 2.053 / 1.763 |
| aso-4 | 1.531 / 1.500 | 1.632 / 1.302 |
| b-tm | 1.309 / 1.302 | 1.430 / 1.426 |
| ytt | 1.862 / 1.781 | 3.814 / 3.056 |
| minoan | 1.772 / 1.668 | 2.434 / 2.128 |

The narrowed candidate scores worse on geographic holdouts in all six cases. The thickness measurements preferentially test locations where a layer was measurable. Angular coverage alone provides insufficient grounds to infer a thin or absent deposit in a blank sector. The case report therefore distinguishes field-supported portions, the measured-fan median, the possible-missing-fan sensitivity, and any plausibility check from an independently investigated archive.

The numeric comparison keeps all estimable held-out folds, including their warnings. An external fold enters weight calibration only when the compared structures jointly passed their sampling checks. Empty or unestimable sectors are retained. When checking a target, the calibration excludes its name, identical input bytes and another record at its recorded vent. The regularization has the weight of one eruption distributed across candidates.

[Matched scores, warnings and search boundaries](matched-holdouts.csv) · [Fold IDs, individual log scores, weights and posterior hashes](../../../src/empirical_isopach/assets/bearing-calibration.json) · [Protocol](../../../docs/methods/archive-evidence.md) · [B-Tm western-sector test](../../../archive/2026-11-19-b-tm-thickness-only/results/directional/bearing/README.md)
