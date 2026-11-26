# Nested isopachs with changing dispersal directions

This study compares positive angular area growth with the preceding dispersal footprints on the same registered observations and buffered geographic partitions. The geography changes within the fitted field: a bounded proximal component, independently variable eccentricity and displacement, positive directional area increments, and a smooth area-preserving angular transformation replace rigid expanding footprints. No plotted contour is smoothed or adjusted independently. [Equations and priors](../../../docs/methods/nested-footprints.md)

The tests include all nine registered eruptions, quadrant omissions, known deposits with narrow and broad unsampled sectors, rotated surveys, and five actual campaign or archive transfers. Measured-site errors, event probabilities for bounds, geographic prediction, geometry and numerical checks are evaluated separately. A better-looking outline does not establish absence in an unsampled direction. These cases informed development; the results are not an external guarantee of calibration.

## Nine-eruption results

The continuous-growth field is the principal nested method. Measured log RMSE improves in **7/9 eruptions**, median absolute log error in 5/9, and the aggregate likelihood of upper limits and ranges in **5/6 bounded datasets**. The clearest geometric gains are coherent turning proximal contours and smoother transitions between dispersal lobes. Aso-4 remains broadly rounded, and several weakly sampled thin margins still expand. [Case-by-case visual and geological findings](visual-review.md) · [Inspected image identities](visual-inspection.json)

Geographic performance remains competitive but does not improve overall: the equal-eruption mean negative log score changes from **1.441 to 1.463**. Three cases improve and six worsen; Mazama has the largest deterioration. On the same buffered folds the GP ensemble averages 1.495 and the spline 1.897. These regional controls predict sites on the same held-out partitions, but their observation-hull volumes are different estimands from a whole-Earth nested volume. Several nominal 95% predictive intervals under-cover the measurements, including Pinatubo at 83.1%.

Values below are previous → revised; lower scores and errors are better. Bound scores average negative log event probabilities, not distances to a limit. All geographic folds, including the preceding numerically uncertain fits, remain included.

| Eruption | Measured log RMSE | Bounded negative log probability | Geographic negative log score | Revised 95% coverage |
|---|---:|---:|---:|---:|
| [Aso-4](visual-review.md#aso-4) | 0.677 → 0.678 | 0.106 → 0.105 | 1.364 → 1.372 | 93.0% |
| [B-Tm](visual-review.md#b-tm) | 0.759 → 0.788 | 1.641 → 0.166 | 1.357 → 1.362 | 94.0% |
| [Ilopango TBJ](visual-review.md#ilopango-tbj) | 1.021 → 1.011 | 1.522 → 1.572 | 1.761 → 1.788 | 89.9% |
| [Mazama](visual-review.md#mazama) | 0.739 → 0.737 | 0.779 → 0.756 | 1.182 → 1.380 | 85.1% |
| [Minoan](visual-review.md#minoan) | 0.989 → 0.984 | — | 1.711 → 1.675 | 90.1% |
| [Pinatubo Layer C](visual-review.md#pinatubo-layer-c) | 0.503 → 0.433 | 1.317 → 1.103 | 1.419 → 1.350 | 83.1% |
| [Sakurajima Satsuma](visual-review.md#sakurajima-satsuma) | 0.497 → 0.483 | 0.258 → 0.220 | 1.092 → 0.963 | 100.0% |
| [Tambora](visual-review.md#tambora) | 0.621 → 0.575 | — | 1.382 → 1.520 | 96.2% |
| [YTT](visual-review.md#ytt) | 0.536 → 0.526 | — | 1.707 → 1.759 | 89.1% |

The final release passes all **9 full fits and 36 geographic refits**, independent draw integration, physical-grid integration and mapped-quartile checks. Reapplying the same full-field population test to the preceding posteriors gives 4/9 passes versus 9/9; the same mapped-quartile check gives 8/9 versus 9/9. The preceding reported geographic fits passed 28/36. Larger populations and exact joint posterior moves resolve numerical disagreement; they do not create evidence for unsampled deposit boundaries. Auxiliary tests retain their separate common ceiling and failures below.

[Measured and bounded agreement](measured-agreement.csv) · [Paired map integrals and geographic scores](paired-results.csv) · [Every geographic prediction](geographic-predictions.csv.gz) · [Sector errors](sector-errors.csv) · [Contour geometry](contour-morphology.csv.gz) · [Matched numerical checks](matched-numerical-checks.json.gz) · [Input-balanced summary](study-summary.json)

## Known deposits and withheld campaigns

The 20 matched input designs include 22 compatible truths, because two observed datasets each admit different hidden deposits. The summary first averages the two spatial scrambles and compatible truths within an input, then gives every input equal weight. Median absolute log error in the integrated median-map volume falls from **0.232 to 0.123**; mean error falls from 0.423 to 0.316. Population agreement improves from 8/20 to 18/20 inputs. The two revised failures are `missing_core`, seed 90, and `missing_sector`, seed 91; their results remain included. Synthetic map recovery uses the same 1,024 posterior draws and two independent spatial scrambles for both methods; these figures are not a claim that the synthetic maps passed the stricter production quartile-field check. [All designs and scores](known-deposits.csv.gz) · [Fit receipts](known-fit-receipts.json.gz)

Volume recovery improves without uniformly better spatial recovery. At 1 cm, the input-balanced median false-positive area grows from about **266,000 to 388,000 km²**; its western part grows from 72,000 to 179,000 km². At 10 cm, median false-positive area falls from 147 km² to zero, while missed area grows from 1,562 to 1,734 km². The narrow-fan and multilobe examples improve substantially, but the two-fan example worsens, and a globally diffuse hidden deposit remains severely underestimated. Identical observations still cannot identify two different hidden truths. These are limits of inference from sparse observations, not evidence for clipping an unsampled sector.

| Withheld campaign | Rows | Previous negative log score ↓ | Area growth ↓ | Numerical qualification |
|---|---:|---:|---:|---|
| Aso-4 | 28 | 1.524 | 1.737 | Both pass; prediction worsens |
| B-Tm | 3 | 3.298 | 2.638 | Revised fit fails the common auxiliary ceiling |
| Calbuco 2015 | 43 | 1.968 | 1.802 | Both pass; prediction improves |
| Historical Pinatubo ten-core preview | 10 | 3.493 | 3.104 | Both fits retain numerical warnings |
| Current Pinatubo marine campaign | 35 | 2.889 | 2.140 | Both fits retain numerical warnings |

Four of five point scores improve, but three revised campaign fits retain numerical warnings. Of the two comparisons with passing fits on both sides, one improves and one worsens. The transfers therefore provide mixed evidence for geographic generalization. All outcomes and actual-site likelihood events remain available in the [campaign table](withheld-surveys.csv.gz) and [receipts](surveys-fit-receipts.json.gz). The 36 requested quadrant omissions produce 33 estimable fits, of which 28 pass; three have insufficient training evidence or an empty withheld sector. [Quadrant scores and warnings](quadrants.csv) · [Partition receipts](quadrant-fit-receipts.json.gz)

## Observation and comparison contract

Each reported minimum is fitted as a point at its reported value, independently of its geological qualifier. Upper limits remain cumulative events and intervals retain both endpoints. Source row IDs, accepted records and hashes are unchanged. No observations, detection limits, zero thicknesses or observation errors are invented. Regional GP and spline controls are rerun on the same accepted rows and geographic folds, with tuning confined to training data.

The actual marine transfer for the current Pinatubo edition contains 35 accepted records, including the newly reviewed WW-SCS-C locality. Its partition is taken from the source ledger rather than the previous edition's 34-row count. The separate ten-core historical preview remains a distinct test. Its 13 source-reported minima are translated into a retained fitting table before fitting either model. An earlier transfer attempt that left them censored is retained as ineligible. Previous and new nested models are refitted to each identical training set.

## Geometry and evidence

Three-scale plates compare the same observations and bounds at proximal, intermediate and full-deposit scales. The final review reads the released median grid used for the volume, with proximal and intermediate frames located in the full view. The review examines curvature, spacing, eccentricity, orientation, connected components, lobe transitions and remaining conditional extrapolation. Contour descriptors diagnose change; they are not fitting targets.

The method retains rounded, single-lobe, turning, overlapping-phase and detached alternatives. Sparse data can still favor rounded contours. The improvement does not enforce a uniformly narrow fan, a preferred eruption volume, or a prescribed reduction in every unsampled sector. The original whole-area sensitivity and preceding fits retain their identities.

## Numerical interpretation

Four independent populations fit both thinning laws. Blocked proposals separate thinning from direction and use exact prior and likelihood ratios. Convergence checks cover intrinsic and continued volume quantiles, training-site medians, and the full mapped geography. All attempts, including failed refinements, are retained. Independent spatial quadrature checks complete-draw volumes; separate refinements check posterior draw count, median-map shape, physical grids and integrated pointwise thickness quartiles.

Production partitions that still fail at 24,576 particles per population receive the same fixed-target Metropolis fallback. Its fixed pilot proposes geometry, thinning law and continuous parameters jointly, using the exact proposal-density correction. It never mixes or resamples states across populations. Selection requires two passing checkpoints, at least 1,024 moves and stability of complete-volume quantiles. This addresses posterior mode movement without changing the scientific model. The known-deposit, quadrant and survey comparisons retain their declared common 12,288-particle ceiling for both methods; their warnings are not replaced by selectively larger fits.

The principal volume integrates the pointwise median thickness over Earth. The reported conditional band integrates the pointwise 25th and 75th thickness percentiles over the same domain. Neither is replaced by the median or quantiles of complete-draw volumes. Numerical agreement is distinct from geological support in unsampled directions.

## Reproduce

Use a fresh ignored destination. The saved fit identities reject altered inputs, engines or run settings; verified checkpoints resume completed attempts.

```sh
python -m benchmarks.regional --output results/nested-adaptive/regional
python -m benchmarks.nested_continuous --output results/nested-adaptive/continuous --workers 4
python -m benchmarks.nested_continuous --design quadrants --output results/nested-adaptive/quadrants --workers 2
python -m benchmarks.nested_growth_challenges --design known --previous-only --output results/nested-adaptive/known
python -m benchmarks.nested_growth_challenges --design known --output results/nested-adaptive/known
python -m benchmarks.nested_growth_challenges --design surveys --output results/nested-adaptive/surveys
python -m benchmarks.nested_growth_baseline --output results/nested-adaptive/baseline-numerics
python -m benchmarks.nested_growth_release --fits results/nested-adaptive/continuous --output results/nested-adaptive/release
python -m benchmarks.nested_growth_review --fits results/nested-adaptive/release --source release --baseline archive/2026-11-25-nested-growth/previous --output results/nested-adaptive/review
python -m benchmarks.nested_growth_study --scratch results/nested-adaptive --output results/nested-adaptive/study
python -m benchmarks.verify_nested_growth --release results/nested-adaptive/release
```

The collector also expects the preceding packets under `results/nested-adaptive/baseline/`, copied from the retained archive. After inspecting every final plate, record the three plate/grid SHA-256 values per case in `visual-inspection.json`, using the published ledger's schema. The collector verifies those inspected bytes against the installed principal maps. It does not infer that an image has been reviewed merely because it exists. [Previous maps and result identities](../../../archive/2026-11-25-nested-growth/README.md) · [Earlier geographic experiments](../geography/README.md) · [Original whole-area study](../whole-area-study.md)
