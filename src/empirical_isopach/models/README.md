# Empirical model families

[Framework](../README.md) · [Method cards](../../../methods/README.md) · [Add a family](../../../docs/ADDING_METHOD.md)

[registry.py](registry.py) is the executable catalogue for the two regional peers:

| Identifier | Implementation |
|---|---|
| `stacked` | [gaussian_process/](gaussian_process/README.md) — one predictive GP ensemble |
| `spline` | [cubic_spline/](cubic_spline/README.md) — one automatically tuned cubic spline |

Each adapter fits only the supplied training observations and returns a surface plus its selection audit. Shared validation, maps, tables and benchmark summaries discover methods from the registry. Display order carries no scientific preference.

The [mass-normalized spherical mixture](mass_mixture/README.md) is a global companion with a separate joint-parameter posterior interface. It uses the same accepted observations and geographic outer folds, but estimates whole-Earth mass directly and includes parameter/prior sensitivity rather than reusing the regional fixed-parameter interval contract.
