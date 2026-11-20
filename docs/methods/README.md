# How the four models fit fallout

An eruption has one accepted table of measured thicknesses and documented bounds. Each of the four models fits that evidence independently. A case page puts their isopachs, bulk volumes and geographic prediction scores side by side. [Eruption catalogue](../../eruptions/README.md) · [Matched comparisons](../../comparisons/README.md)

## Observations and bounds

A confirmed tephra occurrence can constrain a fan edge even when the source gives no bed thickness. Review the original core or section, record what was observed and determine whether it supports a quantitative thickness bound. An accepted bound enters the shared likelihood once. The audit retains the assumptions behind transferred bounds, the grounds for omission, and the weaker or omission comparisons for influential sites. B-Tm works through two lake profiles and a peat mass budget. [Site decision protocol](sites-without-measured-thickness.md) · [Site worksheet](../../templates/SITE_EVIDENCE.md) · [B-Tm source audit](../../data/audits/B-Tm/README.md)

## Regional GP ensemble

The `stacked` model combines directional thinning trends with spatially correlated residual thickness. Predictions at held-out training locations determine the weights of the Gaussian-process candidates. Each simulated deposit uses one candidate over the whole observation hull. The interval conditions on the fitted parameters and weights. [Component equations](gaussian-process.md) · [Ensemble weights and uncertainty](gp-ensemble.md)

## Regional cubic spline

The `spline` model fits a cubic surface to log thickness. Penalties control changes in slope and curvature and shrink the surface toward a thinning trend learned from the training sites. Geographic training folds select the trend, knot spacing and penalty strengths. Integration points evaluate the penalty without adding field observations. Joint coefficient draws give an interval over the same hull as the GP. [Spline equations and tuning](cubic-splines.md) · [Matched regional validation](regional-comparison.md)

## Directional ensemble

The principal map gives the median thickness at each location. Measurements anchor the local interpolated field. Beyond those areas, each simulated deposit can place a fan in a poorly sampled direction. Other eruptions calibrate the weights of six structures spanning three fan widths, with and without an extra fan. The principal bulk volume integrates the median map over the Earth. Integrating each complete deposit first gives a separate volume median and conditional interval. A matched control fixes fan direction with the same data and weights. [Uncertain fan directions](uncertain-azimuth.md) · [Current comparison and known-deposit tests](../../comparisons/directional/README.md)

The measured-fan sensitivity follows sampled bearings. Earlier experiments isolate structure weighting, radial thinning and the proximal cap. [Measured fans and archive evidence](archive-evidence.md) · [Fixed-direction ensemble](directional-ensemble.md) · [Single-structure equations and sensitivities](directional-continuation.md)

## Spherical mixture

A fitted mixture of spatial fans integrates to its volume parameter over the Earth. The reported volume median and conditional interval use complete deposits. Fixed changes to the priors and fan geometry produce a separate sensitivity envelope. B-Tm uses source-focused fans. The catalogue identifies the geometry used in each case. [Likelihood, priors and numerical checks](probabilistic-global.md) · [Source-focused fans](source-anchored-fans.md) · [Six-eruption estimates](../../comparisons/spherical-mixture/README.md)

## Interpreting the maps

The support audit tests where measurements bracket each contour and records holes, rejected bands and partially supported volumes. Each global extrapolation depends on assumptions about unsampled areas. A marine transport and preservation model is documented as a proposed extension. [Support protocol](supported-isopachs.md) · [Marine observation proposal](marine-observations.md)
