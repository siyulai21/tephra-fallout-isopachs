# Empirical model families

[Framework](../README.md) · [Method cards](../../../methods/README.md) · [Add a family](../../../docs/ADDING_METHOD.md)

[registry.py](registry.py) is the executable catalogue. Exactly two families are active:

| Identifier | Implementation |
|---|---|
| `stacked` | [gaussian_process/](gaussian_process/README.md) — one predictive GP ensemble |
| `spline` | [cubic_spline/](cubic_spline/README.md) — one automatically tuned cubic spline |

Each adapter fits only the supplied training observations and returns a surface plus its selection audit. Shared validation, maps, tables and benchmark summaries discover methods from the registry. Display order carries no scientific preference.
