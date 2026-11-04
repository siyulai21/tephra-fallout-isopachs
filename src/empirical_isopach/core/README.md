# Shared statistical and data utilities

[Framework](../README.md) · [Model families](../models/README.md)

| Module | Purpose |
|---|---|
| [observations.py](observations.py) | Strict CSV semantics, row identity, exact values and documented bounds |
| [catalog.py](catalog.py) | The accepted dataset catalogue and installed-data lookup |
| [geometry.py](geometry.py) | Common fitting bounds from sampling coordinates only |
| [validation.py](validation.py) | Geographic folds/buffers and predictive scoring |
| [likelihood.py](likelihood.py) | Stable censored-normal likelihood derivatives shared by the current families |
| [distributions.py](distributions.py) | Gaussian-mixture marginal summaries, quantiles and expected thickness |

Sharing a likelihood or numerical utility does not make GP and spline the same model. A future non-Gaussian model must supply and test its own appropriate distribution interface; it must not be forced into an unjustified Gaussian approximation.
