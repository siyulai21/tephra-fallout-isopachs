# Empirical model families

[Framework](../README.md) · [Method cards](../../../methods/README.md) · [Add a family](../../../docs/ADDING_METHOD.md)

[registry.py](registry.py) is the executable catalogue for the two regional peers:

| Identifier | Implementation |
|---|---|
| `stacked` | [gaussian_process/](gaussian_process/README.md) — one predictive GP ensemble |
| `spline` | [cubic_spline/](cubic_spline/README.md) — one automatically tuned cubic spline |

Each adapter fits only the supplied training observations and returns a surface plus its selection audit. Shared validation, maps, tables and benchmark summaries discover methods from the registry. Display order carries no scientific preference.

The [mass-normalized spherical mixture](mass_mixture/README.md) is a global companion with a separate joint-parameter posterior interface. It uses the same accepted observations and geographic outer folds, but estimates whole-Earth mass directly and includes parameter/prior sensitivity rather than reusing the regional fixed-parameter interval contract.

The [directional continuation](directional/) interpolates latent thickness inside retained measurement triangles and continues from the source across unsampled sectors. The same 13 policies, matched geographic holdouts and quadrant deletions apply to every case. [Equations](../../../docs/methods/directional-continuation.md) · [Results](../../../comparisons/directional/README.md).
