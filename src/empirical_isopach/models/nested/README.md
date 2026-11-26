# Bayesian nested isopachs

The principal method fits positive angular area growth and continuously changing dispersal directions. [Footprint equations and numerical contract](../../../../docs/methods/nested-footprints.md) · [Matched review](../../../../comparisons/nested-isopachs/adaptive-growth/README.md)

The earlier finite-corridor continuation remains a separate sensitivity. [Geographic reach and fitting](reach.py) · [Case reports](reach_report.py) · [Protocol](../../../../docs/methods/nested-reach.md)

The [method equations and reproduction commands](../../../../docs/methods/nested-isopachs.md) describe the prior, external analogue information, observation model, independent particle populations and whole-Earth integration.

| File | Responsibility |
|---|---|
| [footprints/growth.py](footprints/growth.py) | Accumulate positive directional area increments with a bounded proximal component |
| [footprints/flow.py](footprints/flow.py) | Apply and invert the smooth area-preserving change of direction |
| [footprints/continuous.py](footprints/continuous.py) | Fit, identify and read principal continuous-direction posteriors |
| [footprints/flow_inference.py](footprints/flow_inference.py) | Tempered sampling with separate thinning and angular proposal blocks |
| [footprints/numerics.py](footprints/numerics.py) | Stable compiled point, upper-limit and interval likelihoods |
| [footprints/rejuvenation.py](footprints/rejuvenation.py) | Fixed-target joint posterior moves after unsuccessful population refinements |
| [footprints/map_convergence.py](footprints/map_convergence.py) | Refine all three mapped thickness quartiles and their spatial fields |
| [footprints/release.py](footprints/release.py) | Assemble selected fits, retain checkpoints and publish principal maps and diagnostics |
| [geometry.py](geometry.py) | Rank geography by equal-area nested footprints |
| [budget_profile.py](budget_profile.py) | Calibrate relative distal mass and solve for a positive decay scale |
| [continuation.py](continuation.py) | Preserve every training prediction and transfer remaining distal mass at the last reached area rank |
| [profile.py](profile.py) | Invert monotone thickness curves and integrate their volumes |
| [inference.py](inference.py) | Joint Bayesian inference and predictive scores |
| [study.py](study.py) | Check source identities, persist draws and refit buffered folds |
| [diagnostics.py](diagnostics.py) | Audit geographic allocations and predictive intervals |
| [distal.py](distal.py) | Test additional thinning beyond all training-site area ranks and score withheld bounds |
| [bearing_check.py](bearing_check.py) | Compare broad continuation with an uncertain or enhanced hidden fan on the same bearings and training sites |
| [report.py](report.py) | Render isopachs, supported-contour decisions, volume checks, archive probes and reports |
