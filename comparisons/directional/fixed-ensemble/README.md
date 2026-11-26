# Directional controls on the current eruption data

Each row uses the same input and buffered geographic holdouts as its eruption page. The single-structure fit uses the reference fan geometry. The fixed-direction ensemble averages six structures. The measured-fan column gives the quantitative-only bearing fit where that diagnostic was published. Lower scores indicate better predictions. [Principal uncertain-direction fits](../README.md) · [Model definitions](../../../docs/methods/directional-ensemble.md)

| Eruption | Sites | Single-structure score | Fixed-direction score | Measured-fan score | Fixed median map (km³) |
|---|---:|---:|---:|---:|---:|
| [Minoan](../../../eruptions/minoan/results/minimum-fit-2026-11-25/directional/ensemble/README.md) | 101 | 1.857 | 1.820 | 1.902 | 35.17 |
| [Sakurajima Satsuma (P14)](../../../eruptions/sakurajima-satsuma/results/minimum-fit-2026-11-25/directional/ensemble/README.md) | 57 | 1.289 | 1.127 † | 1.295 | 8.31 |
| [Tambora](../../../eruptions/tambora/results/directional/ensemble/README.md) | 26 | 1.953 | 1.964 † | — | 105.80 |
| [YTT](../../../eruptions/ytt/results/directional/ensemble/README.md) | 46 | 1.811 | 1.777 † | — | 985.66 |

A dagger marks a numerical warning on the fixed ensemble’s geographic fits or its median-map integral. Each case links the individual policies, holdouts and sampling diagnostics.

![Whole-Earth volumes of the fixed-direction median maps](volumes.png)

## Known deposits and calibration

The 14 fixed-design trials compare fitted isopach area with known deposits at 1, 10 and 100 cm. Two trials have identical observed thicknesses under two different hidden deposits. Their fitted intervals covered the larger hidden deposit in neither trial. [Known volumes](known-volumes.csv) · [Contour overlap](contour-accuracy.csv) · [Simulation details](simulation-records.json.gz)

The original six-eruption training packet fixes the structure weights for these experiments. A target eruption contributes no scores to its own calibration. [Training inputs and weights](../../../src/empirical_isopach/assets/directional-calibration.json) · [Independent numerical checks](verification.json)

The current principal packets for Aso-4, B-Tm, Ilopango TBJ (total primary airfall), Mazama, Pinatubo 1991 (Layer C) do not contain these older fixed and single-structure controls. They are omitted from this matched control table; their current directional reconstructions remain in the principal comparison. Earlier study controls retain their original input identities.
