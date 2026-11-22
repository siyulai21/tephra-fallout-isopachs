# Pinatubo before the Koyaguchi visual review

This snapshot records the 194-site input and all five fits before four C1 readings were changed on 22 November 2026. It contains 140 measured thicknesses, 26 lower bounds and 28 upper bounds. Its accepted-input SHA-256 is `5400069fea176a055aebf2395d0f92dabf51ce3c7354f17c559f4e9ccea0bfda`. [Original input](input.csv) · [Original sublayer transcription](koyaguchi-figure-3.csv) · [Original Figure 7 transcription](paladio-figure-7.csv) · [Site selection](selection.csv) · [Source hashes and coverage](provenance.json)

| Method | Domain | Median-map bulk volume (km³) | Conditional 95% interval (km³) |
|---|---|---:|---:|
| [GP](results/report.json) | Observation hull | 3.50 | 3.00–5.05 |
| [Cubic spline](results/report.json) | Observation hull | 13.17 | 11.28–2.43 × 10¹¹ |
| [Directional ensemble](results/directional/azimuth/report.json) | Whole Earth | 6.57 | 5.00–11.76 |
| [Spherical mixture](results/spherical-mixture/report.json) | Whole Earth | 13.70 | 8.79–25.38 |
| [Nested corridors](results/nested-isopachs/report.json) | Whole Earth | 7.58 | 5.70–9.08 |

The spline's upper conditional volume failed its integration check. Two of six directional ensemble members carried sampling warnings. The spherical reference fit had one divergence. The nested full fit passed its checks, while two of four geographic training fits passed. The exact sites beyond 100 km left a 294° gap in bearing. [Full precision, geographic scores and numerical flags](volumes.csv)

The [archived results folder](results/README.md) and the frozen GitHub revision preserve all 227 result files, including three global median fields, both regional fits, geographic holdouts, prior sensitivities and source checks. All numerical files and figures are byte-identical to their originals. One archived Markdown page points its cross-repository links to the frozen revision. The checksums verify every original at its path in that revision. [Frozen case report and maps](https://github.com/siyulai21/tephra-fallout-isopachs/blob/38c5769318a6449128f6a6198e739a8fc34829dc/eruptions/pinatubo-1991-layer-c/README.md) · [Complete result files](https://github.com/siyulai21/tephra-fallout-isopachs/tree/38c5769318a6449128f6a6198e739a8fc34829dc/eruptions/pinatubo-1991-layer-c/results) · [227 file checksums](results.sha256)

The later Koyaguchi review retained the same site list and revised four terrestrial thicknesses. [Reviewed readings and original pixels](../../../data/audits/Pinatubo-1991/koyaguchi-c1-review.md)
