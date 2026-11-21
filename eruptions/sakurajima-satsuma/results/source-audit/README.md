# Sensitivity to the western marine core

The 8 mm Satsuma bed in East China Sea core MD98-2195 is the farthest accepted thickness, 163 km west of the summit reference. Moriwaki et al. describe a lenticular vitric ash bed. A lens can preserve a local thickness different from the regional fallout. The main fit includes this directly measured and geochemically correlated bed. A separate calculation removes that one row and refits both global families on the same remaining 56 observations. [Original description](https://doi.org/10.1016/j.quaint.2011.03.046) · [Audit and site distances](../../../../data/audits/Sakurajima-Satsuma/README.md)

| Estimate | All sites | Without MD98-2195 |
|---|---:|---:|
| Directional median-map integral (km³) | 7.08 | 9.64 |
| Directional whole-deposit median [conditional 95%], km³ | 10.0 [3.8–70.6] | 15.0 [4.4–168.2] |
| Spherical whole-deposit median [conditional 95%], km³ | 11.3 [5.1–666.5] | 54.7 [5.5–6,801.9] |

With the core withheld, the directional model predicts 0.59 [3.3351e-05–12.2] cm at the core, and the spherical model predicts 2.37 [0.11413–41.8] cm. Brackets give 95% observation-predictive intervals. Both intervals include the measured 0.8 cm. The broad intervals admit many other thicknesses at this distance.

Removing the core raises the spherical whole-deposit median by a factor of 4.8 and the directional median-map integral by a factor of 1.36. Without it, the remaining western measurements are 70 cm at 18 km and 15 cm at 32 km. The western distal continuation then has little quantitative constraint. A sensitivity to deletion measures the leverage of the bed, and does not estimate redistribution or demonstrate that the bed is biased.

The refits retain the same six directional structures, external calibration, common continuation assumptions and spherical baseline prior. Each family rebuilds its geometry on the retained observations. The spherical sampler, spatial integration and draw-split checks passed in the omission fit. The directional omission passed the map, joint-volume split and integration checks. Each report contains the row IDs and fit diagnostics. [Directional refit](omit-marine-directional.json) · [Spherical refit](omit-marine-spherical.json)

From the repository root, run `python -m benchmarks.satsuma_core_sensitivity --verify` to check the published records. To repeat the fits, run `python -m benchmarks.satsuma_core_sensitivity directional --output results/satsuma-core-check` and repeat with `spherical`.
