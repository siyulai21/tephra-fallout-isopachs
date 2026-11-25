# How the five models fit fallout

An eruption has one accepted table of measured thicknesses and documented bounds. Each of the five models fits that evidence independently. A case page puts their isopachs, bulk volumes and geographic prediction scores side by side. [Eruption catalogue](../../eruptions/README.md) · [Matched comparisons](../../comparisons/README.md)

Each global family now limits extrapolation according to the sampled directions and retains its earlier field as a sensitivity. [Current continuation rules](finite-spread.md) · [Nine-eruption tests](../../comparisons/global-extent/README.md)

## Observations and bounds

A confirmed tephra occurrence can constrain a fan edge even when the source gives no bed thickness. Review the original core or section, record what was observed and determine whether it supports a quantitative thickness bound. An accepted bound enters the shared likelihood once. The audit retains the assumptions behind transferred bounds, the grounds for omission, and the weaker or omission comparisons for influential sites. B-Tm works through two lake profiles and a peat mass budget. [Site decision protocol](sites-without-measured-thickness.md) · [Site worksheet](../../templates/SITE_EVIDENCE.md) · [B-Tm source audit](../../data/audits/B-Tm/README.md)

## Regional GP ensemble

The `stacked` model combines directional thinning trends with spatially correlated residual thickness. Predictions at held-out training locations determine the weights of the Gaussian-process candidates. Each simulated deposit uses one candidate over the whole observation hull. The interval conditions on the fitted parameters and weights. [Component equations](gaussian-process.md) · [Ensemble weights and uncertainty](gp-ensemble.md)

## Regional cubic spline

The `spline` model fits a cubic surface to log thickness. Penalties control changes in slope and curvature and shrink the surface toward a thinning trend learned from the training sites. Geographic training folds select the trend, knot spacing and penalty strengths. Integration points evaluate the penalty without adding field observations. Joint coefficient draws give an interval over the same hull as the GP. [Spline equations and tuning](cubic-splines.md) · [Matched regional validation](regional-comparison.md)

## Directional ensemble

The principal map integrates uncertain fan headings, broad thinning and local departures supported by neighboring sites. The fan can turn or split as distance increases. Each observation contributes its recorded measurement, upper or lower limit, or interval. Integrating the median-thickness surface over Earth gives the principal volume. An additional interval comes from coherent whole-deposit draws. [Field, likelihood and numerical checks](directional-multiscale.md) · [Three-scale maps and matched predictions](../../comparisons/directional/multiscale/README.md)

The measured-fan sensitivity follows sampled bearings. Earlier experiments isolate structure weighting, radial thinning and the proximal cap. [Measured fans and archive evidence](archive-evidence.md) · [Fixed-direction ensemble](directional-ensemble.md) · [Single-structure equations and sensitivities](directional-continuation.md)

## Spherical mixture

The principal spherical field integrates the retained kernel mass along measured and uncertain bearings over the Earth. The reported volume integrates the median-thickness map over Earth. The conditional interval comes from complete posterior deposits. Fixed changes to the priors and fan geometry produce a separate sensitivity envelope. B-Tm uses source-focused fans. The catalogue identifies the geometry used in each case. [Finite bearing reach and uncertain continuation](spherical-bearings.md) · [Likelihood, priors and numerical checks](probabilistic-global.md) · [Source-focused fans](source-anchored-fans.md) · [Estimates by eruption](../../comparisons/spherical-mixture/README.md)

## Bayesian nested isopachs

The nested model infers the direction, elongation and reach of source-connected phases, with an alternative for a detached maximum. A common proximal deposit can feed a shorter or longer secondary lobe. The principal volume integrates pointwise median thickness over Earth. Each case retains the original whole-area sensitivity. [Dispersal footprints and limits](nested-footprints.md) · [Underlying thinning model](nested-isopachs.md) · [Matched geography and volume tests](../../comparisons/nested-isopachs/geography/README.md)

## Interpreting the maps

The support audit tests where measurements bracket each contour and records holes, rejected bands and partially supported volumes. Each global extrapolation depends on assumptions about unsampled areas. A marine transport and preservation model is documented as a proposed extension. [Support protocol](supported-isopachs.md) · [Marine observation proposal](marine-observations.md)
