# Directional controls on the current eruption data

Each row uses the same input and buffered geographic holdouts as its eruption page. The single-structure fit uses the reference fan geometry. The fixed-direction ensemble averages six structures. The measured-fan column gives the quantitative-only bearing fit where that diagnostic was published. Lower scores indicate better predictions. [Principal uncertain-direction fits](../README.md) · [Model definitions](../../../docs/methods/directional-ensemble.md)

| Eruption | Sites | Single-structure score | Fixed-direction score | Measured-fan score | Fixed median map (km³) |
|---|---:|---:|---:|---:|---:|
| [Aso-4](../../../eruptions/aso-4/results/directional/ensemble/README.md) | 87 | 1.555 | 1.507 | — | 739.97 |
| [B-Tm](../../../eruptions/b-tm/results/directional/ensemble/README.md) | 71 | 1.274 | 1.266 † | 1.244 | 33.92 |
| [Ilopango TBJ (total primary airfall)](../../../eruptions/ilopango-tbj/results/directional/ensemble/README.md) | 168 | 1.328 | 1.252 | 1.361 | 71.11 |
| [Mazama](../../../eruptions/mazama/results/directional/ensemble/README.md) | 300 | 1.002 | 0.781 † | 0.898 | 306.79 |
| [Minoan](../../../eruptions/minoan/results/directional/ensemble/README.md) | 101 | 1.732 | 1.695 | — | 33.67 |
| [Sakurajima Satsuma (P14)](../../../eruptions/sakurajima-satsuma/results/directional/ensemble/README.md) | 57 | 1.264 | 1.069 † | 1.244 | 8.57 |
| [Tambora](../../../eruptions/tambora/results/directional/ensemble/README.md) | 26 | 1.953 | 1.964 † | — | 105.80 |
| [YTT](../../../eruptions/ytt/results/directional/ensemble/README.md) | 46 | 1.811 | 1.777 † | — | 985.66 |

A dagger marks a numerical warning on the fixed ensemble’s geographic fits or its median-map integral. Each case links the individual policies, holdouts and sampling diagnostics.

![Whole-Earth volumes of the fixed-direction median maps](volumes.png)

## Known deposits and calibration

The 14 fixed-design trials compare fitted isopach area with known deposits at 1, 10 and 100 cm. Two trials have identical observed thicknesses under two different hidden deposits. Their fitted intervals covered the larger hidden deposit in neither trial. [Known volumes](known-volumes.csv) · [Contour overlap](contour-accuracy.csv) · [Simulation details](simulation-records.json.gz)

The original six-eruption training packet fixes the structure weights for these experiments. A target eruption contributes no scores to its own calibration. [Training inputs and weights](../../../src/empirical_isopach/assets/directional-calibration.json) · [Independent numerical checks](verification.json)
