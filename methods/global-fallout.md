# Conditional global fallout model

[Method catalogue](README.md) · [Global comparison](../comparisons/global/README.md) · [Output definitions](../docs/OUTPUTS.md)

**Public command: `global`; also included by `compare`.** This companion fits a positive field of jointly estimated fallout fans with total mass normalized over the whole Earth. Its global volume is independent of a chosen map rectangle or a closing isopach. It uses the same accepted observations and geographic holdouts as the regional peers.

The current catalogue chooses the reviewed geometry: B-Tm uses source-focused fans; the other cases use displaced components. Aso-4's high global total remains questioned. The tighter fan-position prior tested in its evidence study is a development alternative, not the universal default. Source-envelope geometries remain experimental.

Posterior draws include parameter uncertainty within each model and prior. Fixed sensitivity refits expose dependence on generic assumptions; their envelope is not a credible interval. Neither numerical convergence nor a finite Earth integral establishes that sparse observations identify unseen mass. Source-focused shape constraints are assumptions, not artificial source-thickness measurements.

The global model has a separate posterior interface under `models/mass_mixture/`. It is not another entry in the regional fixed-parameter registry. A failed local contour screen remains an evidence warning and does not veto the conditional global calculation.

[Equations and priors](../docs/methods/probabilistic-global.md) · [Source-focused geometry](../docs/methods/source-anchored-fans.md) · [Implementation](../src/empirical_isopach/models/mass_mixture/README.md) · [Known-truth and development evidence](../docs/VALIDATION.md)
