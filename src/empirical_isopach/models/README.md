# Empirical model families

[Framework](../README.md) · [Method cards](../../../docs/methods/README.md) · [Add a family](../../../docs/ADDING_METHOD.md)

[registry.py](registry.py) is the executable catalogue for the two regional peers:

| Identifier | Implementation |
|---|---|
| `stacked` | [gaussian_process/](gaussian_process/README.md) — one predictive GP ensemble |
| `spline` | [cubic_spline/](cubic_spline/README.md) — one automatically tuned cubic spline |

Each adapter fits only the supplied training observations and returns a surface plus its selection audit. Shared validation, maps, tables and benchmark summaries discover methods from the registry. Display order carries no scientific preference.

The [mass-normalized spherical mixture](mass_mixture/README.md) is a global companion with a separate joint-parameter posterior interface. It uses the same accepted observations and geographic outer folds, but estimates whole-Earth mass directly and includes parameter/prior sensitivity rather than reusing the regional fixed-parameter interval contract.

The [directional ensemble](directional/README.md) fits uncertain fan directions, thinning and spatially correlated local departures. Read `directional/deposit.json` for the whole-Earth median-map integral and `directional/multiscale/report.json` for complete-field volume uncertainty and geographic tests. The 13-policy single-structure study remains available for sensitivity tests. [Principal protocol](../../../docs/methods/directional-multiscale.md) · [Directional validation](../../../comparisons/directional/README.md)

[Bayesian nested isopachs](nested/README.md) jointly fit geographic contour ranks and an area–thickness curve. Each draw has a closed-form whole-Earth volume. Each case includes a geographic quadrature check, thickness-slice budgets and training-only distal sensitivities. [Nested method](../../../docs/methods/nested-isopachs.md)
