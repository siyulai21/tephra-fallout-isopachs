# Bayesian nested isopachs

The full comparison also runs the paired finite-corridor continuation. [Geographic reach and fitting](reach.py) · [Case reports](reach_report.py) · [Protocol](../../../../docs/methods/nested-reach.md)

The [method equations and reproduction commands](../../../../docs/methods/nested-isopachs.md) describe the prior, external analogue information, observation model, independent particle populations and whole-Earth integration.

| File | Responsibility |
|---|---|
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
