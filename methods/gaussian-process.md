# GP ensemble

[Active methods](README.md) · [Regional evidence](../comparisons/regional/README.md)

**Public identifier: `stacked`.** A directional thinning trend describes broad variation in log thickness; a Matérn Gaussian process describes spatial residuals. Geographic training predictions determine weights across candidate trends, length scales and noise settings. One fitted ensemble is reported for each eruption.

A single selected GP remains a component-level/historical option. It is not a separate active method. The ensemble uses no cubic-spline basis or spline knots.

Mixture quantiles and expected thickness use the full predictive mixture. Volume draws select one component for each whole correlated field, retaining between-component differences. Parameters and weights remain fixed in those draws; conditional intervals omit their estimation uncertainty.

[Model equations](../docs/methods/gaussian-process.md) · [Stacking equations and limitations](../docs/methods/gp-ensemble.md) · [Shared regional evaluation](../docs/methods/regional-comparison.md) · [Implementation](../src/empirical_isopach/models/gaussian_process/README.md)
