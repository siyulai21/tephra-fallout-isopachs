# Pinatubo source sensitivities

The original measurements support three checks on the nested-corridor estimate and a matched upper-endpoint check on both regional models. All fits use the same vent and model policies. Each fits its own training selection. The numbers are whole-Earth integrals of median thickness and conditional 95% intervals for coherent bulk-volume draws.

| Fit | Training rows | Estimated volume (km³) | Conditional 95% (km³) | Exact withheld beds inside 95% | Numerical checks |
|---|---:|---:|---:|---:|---|
| [Common five-family input](../nested-isopachs/README.md) | 193 | 7.75 | 7.01–9.35 | 84/139 over four geographic folds | Full fit passed, 1/4 geographic fits passed |
| [Withhold all offshore core evidence](marine-holdout/result.json) | 160 | 0.70 | 0.61–1.56 | [0/18](marine-holdout/heldout.csv) | Parent populations disagreed |
| [Withhold numerical readings from the two land maps](tabulated-thickness/result.json) | 47 | 7.73 | 6.47–12.10 | [100/109](tabulated-thickness/heldout.csv) | Importance weights failed |
| [Use both endpoints of nine marine replicate ranges](marine-intervals/result.json) | 193 | 5.42 | 4.48–7.96 | — | Passed |

The land-only model systematically missed the offshore deposit. The principal survey continues hundreds of kilometres into the South China Sea, where later cores record a secondary thickness maximum. Assigning an unsurveyed distal sector a thin deposit can give an artificially small volume. The land-only volume also carries a population warning. [Predictions at all 33 offshore records](marine-holdout/heldout.csv) · [Wiesner et al.](https://doi.org/10.1007/s00445-003-0306-x)

Table-based thicknesses retain 47 observations, including named terrestrial sections located from their map. The 146 withheld rows are the numerical Figure 7 and C1/C2 map readings. The interval is wider, and the alternative fit failed its importance-weight check. Its central value alone gives little evidence of robustness. [All withheld map readings](tabulated-thickness/heldout.csv)

Nine Hess rows represent ranges between replicate observations. The common input takes the lower endpoint as a minimum because all five families can fit that evidence. The interval experiment gives each location its full reported bracket. This tests a different interpretation of variation within a station group. It gives a materially smaller inferred deposit and is explicit model uncertainty beyond any single reported interval. [Both endpoints at each station](marine-intervals/interval-input.csv) · [Marine reading guide](../../../../data/audits/Pinatubo-1991/marine-check.md)

## What the spline predicts at an actual core

At H01-16 (SO-114-13) the source gives a 1–2 mm replicate range. The common input uses ≥0.1 cm. The spline predicts 3,199 cm at that coordinate and the GP predicts 0.49 cm. A northern lobe is visible along the hull boundary. The nearest other exact reading to its plotted grid maximum is about 117 km away. Every conditional mean inside the exported hull grid is finite, although the spline overflows at 71 grid cells outside it. The spline’s median integral and conditional interval both carry numerical warnings. [Raw peak and site checks](regional-peak.json) · [Source range](../../../../data/audits/Pinatubo-1991/marine-check.md)

Using the other endpoint as an upper limit at the same nine sites changes the GP hull estimate from 3.51 to 3.21 km³ and the spline from 14.07 to 5.75 km³. The spline still has an extremely long upper volume tail. Both fits use the same coordinates, common hull and geographic partitions. The two endpoint fits illustrate information lost when a two-sided reading must be represented by one inequality. Their difference is a sensitivity result. [Matched regional refits](regional-upper-endpoints/regional/README.md) · [Upper-endpoint input](regional-upper-endpoints/upper-endpoints.csv) · [Median thickness from every fitted family at all nine stations](range-predictions.csv)

Reproduce all four checks with `python -m benchmarks.pinatubo_sensitivity --output results/pinatubo-source-tests`. Each numerical report includes training and withheld row IDs, population checks, quadrature history and the posterior packet hash.
