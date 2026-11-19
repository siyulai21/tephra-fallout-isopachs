# Directional fallout implementation

Start with `directional_dataset` in [reporting.py](reporting.py). It runs the common field fits, the six-structure ensemble, the measured-fan sensitivity and the principal uncertain-direction reconstruction. The normal case command publishes one family report in `results/directional/README.md`.

| Work on | Start here |
|---|---|
| Principal field, leave-one-eruption-out weights and whole-Earth integrals | [Azimuth field](azimuth/field.py) · [Calibration](azimuth/calibration.py) · [Integrals](azimuth/integrals.py) |
| Principal fit, held-out predictions, maps and family report | [Run and validate](azimuth/report.py) · [Replay the median field](azimuth/primary.py) · [Write the narrative](azimuth/narrative.py) |
| The six base structures and their earlier fixed-direction calibration | [Ensemble members](ensemble/members.py) · [Weight calibration](ensemble/weights.py) · [Fixed-direction report](ensemble/report.py) |
| Measured-fan reach and evidence from inspected sites | [Measured-fan runs](bearing_report.py) · [Inventory and visibility assumptions](archive_readings.py) · [Condition the principal field](azimuth/archive.py) |
| Underlying field and single-structure assumptions | [Geometry](geometry.py) · [Inference](inference.py) · [Fits and retries](study.py) · [Whole-Earth integration](integration.py) · [Unobserved proximal cap](source_cap.py) |

The local field interpolates log thickness on retained measurement triangles. Each base structure continues fallout across gaps and beyond the sampled area. The principal fit varies the direction of extra fallout across coherent simulated deposits. Its paired control holds directions fixed under the same weights. The older fixed-direction ensemble uses its separately fitted weights. [Method and map interpretation](../../../../methods/directional-fallout.md)

Run `python -m benchmarks.verify_azimuth_continuation` for the principal reconstruction. The study index links the paired results, the earlier calibrations and the known-deposit trials. [Directional studies](../../../../comparisons/directional/README.md) · [Verification commands](../../../../benchmarks/README.md)
