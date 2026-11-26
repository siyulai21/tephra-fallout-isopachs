# Nested isopachs with changing dispersal directions

This study compares positive angular area growth with the preceding dispersal footprints on the same registered observations and buffered geographic partitions. The geography changes within the fitted field: a bounded proximal component, independently variable eccentricity and displacement, positive directional area increments, and a smooth area-preserving angular transformation replace rigid expanding footprints. No plotted contour is smoothed or adjusted independently. [Equations and priors](../../../docs/methods/nested-footprints.md)

The tests include all nine registered eruptions, quadrant omissions, known deposits with narrow and broad unsampled sectors, rotated surveys, and five actual campaign or archive transfers. Measured-site errors, event probabilities for bounds, geographic prediction, geometry and numerical checks are evaluated separately. A better-looking outline does not establish absence in an unsampled direction. These cases informed development; the results are not an external guarantee of calibration.

## Observation and comparison contract

Each reported minimum is fitted as a point at its reported value, independently of its geological qualifier. Upper limits remain cumulative events and intervals retain both endpoints. Source row IDs, accepted records and hashes are unchanged. No observations, detection limits, zero thicknesses or observation errors are invented. Regional GP and spline controls are rerun on the same accepted rows and geographic folds, with tuning confined to training data.

The actual marine transfer for the current Pinatubo edition contains 35 accepted records, including the newly reviewed WW-SCS-C locality. Its partition is taken from the source ledger rather than the previous edition's 34-row count. The separate ten-core historical preview remains a distinct test. Previous and new nested models are refitted to each identical training set.

## Geometry and evidence

Three-scale plates compare the same observations and bounds at proximal, intermediate and full-deposit scales. The final review reads the released median grid used for the volume, with proximal and intermediate frames located in the full view. The review examines curvature, spacing, eccentricity, orientation, connected components, lobe transitions and remaining conditional extrapolation. Contour descriptors diagnose change; they are not fitting targets.

The method retains rounded, single-lobe, turning, overlapping-phase and detached alternatives. Sparse data can still favor rounded contours. The improvement does not enforce a uniformly narrow fan, a preferred eruption volume, or a prescribed reduction in every unsampled sector. The original whole-area sensitivity and preceding fits retain their identities.

## Numerical interpretation

Four independent populations fit both thinning laws. Blocked proposals separate thinning from direction and use exact prior and likelihood ratios. Convergence checks cover intrinsic and continued volume quantiles, training-site medians, and the full mapped geography. All attempts, including failed refinements, are retained. Independent spatial quadrature checks complete-draw volumes; separate refinements check posterior draw count, median-map shape, physical grids and integrated pointwise thickness quartiles.

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
python -m benchmarks.nested_growth_release --fits results/nested-adaptive/continuous --output results/nested-adaptive/release
python -m benchmarks.nested_growth_review --fits results/nested-adaptive/release --source release --baseline archive/2026-11-25-nested-growth/previous --output results/nested-adaptive/review
python -m benchmarks.nested_growth_study --scratch results/nested-adaptive --output results/nested-adaptive/study
python -m benchmarks.verify_nested_growth --release results/nested-adaptive/release
```

The collector also expects the preceding packets under `results/nested-adaptive/baseline/`, copied from the retained archive. [Previous maps and result identities](../../../archive/2026-11-25-nested-growth/README.md) · [Earlier geographic experiments](../geography/README.md) · [Original whole-area study](../whole-area-study.md)
