# Directional fallout implementation

The assistant-led eruption command fits the multiscale directional ensemble and writes its median isopachs and whole-Earth integral at the family entry point. The older directional fits supply matched sensitivity controls. [Entry point](reporting.py) · [Equations, limits and numerical checks](../../../../docs/methods/directional-multiscale.md)

| Calculation | Implementation |
|---|---|
| Evolving fan headings, thinning and physical correlation lengths | [Field geometry](multiscale/geometry.py) · [Training-only directional reach](multiscale/spread.py) |
| Measurements, bounds, posterior weights and coherent draws | [Censored likelihood correction](multiscale/conditioning.py) · [Inference and checks](multiscale/fit.py) · [Field replay](multiscale/posterior.py) |
| Spatial holdouts, integration, archive probes and support | [Deleted regions](multiscale/validation.py) · [Whole-deposit volumes](multiscale/integrals.py) · [Publication](multiscale/release.py) |
| Earlier uncertain-azimuth and finite-spread controls | [Azimuth field](azimuth/field.py) · [Calibration](azimuth/calibration.py) · [Integration](azimuth/integrals.py) |
| Six-structure fixed-direction and measured-fan controls | [Ensemble members](ensemble/members.py) · [Calibration](ensemble/weights.py) · [Measured-fan runs](bearing_report.py) · [Archive evidence](archive_readings.py) |
| Earlier single-structure model | [Geometry](geometry.py) · [Inference](inference.py) · [Fits and retries](study.py) · [Proximal cap](source_cap.py) |

Run `python -m benchmarks.verify_directional_multiscale` to replay current fits, held-out event probabilities, whole-Earth volumes and support decisions. Run `python -m benchmarks.verify_azimuth_continuation` for the earlier comparison. [Visual review and known-deposit tests](../../../../comparisons/directional/multiscale/README.md)
