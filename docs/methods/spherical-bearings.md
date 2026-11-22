# Spherical mixture with finite directional reach

The principal spherical result uses the existing Gaussian, exponential and power-law kernels, their inferred scales, offsets or source-focused axes, and their original bounded observation likelihood. It changes the continuation of those kernels into bearings lacking comparable sampling. Each eruption uses the same reach rule and the same distribution of possible hidden fan widths and orientations. [Eruption and known-deposit comparisons](../../comparisons/spherical-mixture/README.md)

## What the observations identify

Field surveys commonly leave sectors and distal limits open. The 1 and 2 cm isopachs of the 1991 Pinatubo marine fallout illustrate how the locations of cores affect inferred contour extent. Fitted thickness and extrapolated volume can also respond differently to the same isopach uncertainty. These observations motivate comparing continuations. They do not supply numerical wind directions or widths for this model. [Wiesner et al., Pinatubo marine cores](https://pubs.usgs.gov/pinatubo/wiesner/) · [Yang and Jenkins, isopach-volume uncertainty](https://doi.org/10.1007/s00445-023-01652-1)

Let the parent spherical mixture have thickness `T₀(s) = 10⁵ V₀ ∑ wⱼ Kⱼ(s)` in centimetres, with each kernel integrating to one per square kilometre on the WGS84 authalic sphere. The catalogue retains its reviewed baseline geometry, including source-focused fans for B-Tm. All phases of the Minoan fallout enter a single likelihood. [Kernel equations and hyperpriors](probabilistic-global.md) · [Source-focused spherical geometry](source-anchored-fans.md)

## Directional reach from the training design

Use finite source-centred authalic chord distances. For exact and lower-bound sites with radii `rᵢ` and bearings `θᵢ`, define the directional reach

```text
R(θ) = maxᵢ rᵢ max(0, cos(θ − θᵢ))ᵇ
b    = log(1/2) / log(cos γ),       γ = 22.5°
r'   = min(r, maxᵢ rᵢ)
d    = max(0, r' − R(θ))
h₀   = d² / [d² + s² + (r' sin γ)²]
```

Here `s` is the median nearest-neighbour distance among distinct exact sites. The reach of an individual ray halves at γ. A proximal outcrop licenses little distal reach on its bearing. Capping the comparison radius at the largest sampled positive radius distinguishes a missing direction from radial thinning beyond the survey in a measured direction. The parent kernel continues to determine the latter. Repeating the baseline with γ of 11.25° and 45° gives a scale sensitivity for each eruption.

Multiply `h₀` by a smooth transition from zero to one outside the retained exact-site interpolation triangles, and another around the nearest upper-limit site. Each uses `t²(3−2t)` with `t = min(distance / s, 1)`. The footprint distance uses kilometres in the equal-area map and the upper-site distance uses the authalic chart. An upper limit anchors the original local likelihood without adding a positive-bearing ray. The final exposure `h(s)` is zero at the source, at every training site and within the retained triangles. A low value along a ray indicates an assumption of directional continuity. The separate support audit retains actual holes and every contour rejection.

## A coherent continuation of the same kernels

Each complete deposit draws an orientation `Φ` uniformly around the source. Independently draw `p` from 0, 1 and 4 with equal probability. Use a single pair across the entire deposit. Set

```text
F₀(α) = 1
Fₚ(α) = [(1 + cos α)/2]ᵖ
B(s)  = 1 − h(s) + h(s) Fₚ(θ − Φ)
T(s)  = T₀(s) B(s)
Zⱼ    = ∫ Kⱼ(s) B(s) dA
V     = V₀ ∑ wⱼ Zⱼ
```

The original broad continuation is possible in one third of deposits. The other two have half-thickness angles of 90° and about 47° under full exposure. The equal weights and widths are common structural assumptions fixed before the benchmark. No case chooses the width that minimizes its volume. A poorly sampled direction can carry a real fan.

The revised field is still a mass-normalized spherical mixture. Its component kernels are `Kⱼ B / Zⱼ`, with weights `wⱼ Zⱼ / ∑ wₖ Zₖ` and total volume `V`. This preserves the integrated retained mass. The principal assumption `0 < B ≤ 1` can only reduce or preserve thickness relative to the same parent draw. This is a physical limitation of the chosen continuation. A separate hidden-fan sensitivity divides `F₄` by its azimuthal mean `70/256`, allowing the hidden axis to exceed the parent thickness. That sensitivity preserves thickness in expectation over orientations at each point. The source-focused principal reconstruction preserves the parent maximum at the volcano, although transitions near upper bounds can interrupt radial monotonicity. The displaced parent still allows genuine secondary maxima.

All accepted thicknesses and bound events see exactly the parent likelihood since `B=1` at their coordinates. Verified parent posterior samples can therefore be reused *exactly for the parameters they describe*, conditional on the sampling design. The extended prior is specified on the parent amplitude `V₀`, kernel parameters, `p` and `Φ`. It induces a design-dependent prior on the derived total `V`. Treating the old independent volume prior as a prior on `V` itself would define a different model and require refitting. Survey location choices and deposit preservation are not modelled as a joint observation process.

## Integration, validation and reproducibility

The primary volume integrates the mapped pointwise median thickness over the entire Earth. Positive quadrature weights integrate each spatially correlated draw. The 2.5th and 97.5th percentiles give the conditional 95% volume interval. The saved posterior and bearing orientations support both summaries. [Map integral and replay checks](../OUTPUTS.md#read-the-map-and-its-volume)

All seven prior alternatives reuse their own checked posterior fits. Each of four buffered geographic folds uses only its training locations and its verified training posterior. Site scores integrate the narrow and contaminated observation distributions, with one-sided probabilities for bounds. The study reports both false extension and missed fallout on 24 existing known-volume fits, including cases with entire sectors missing, invisible proximal caps and diffuse global tails. Some experiments retain unresolved contour areas or failed numerical checks. The real geographic tests do not observe fallout in an entirely unvisited sector.

```sh
tephra-isopachs global Aso-4 --output results/aso-spherical
tephra-isopachs compare path/to/eruption.csv --output results/new-eruption
python -m benchmarks.global_posterior --summarize-only --published --output results/spherical-comparison
python -m benchmarks.verify_spherical_bearings
```

To repeat the known-volume fits from their original seeds and score the alternative continuations, use fresh directories for each geometry. Each score uses the parameter file named and hashed by that fit's own report.

```sh
python -m benchmarks.mass_mixture --output results/displaced-stress
python -m benchmarks.source_fans --output results/source-stress
python -m benchmarks.source_fans --summarize-only --output results/source-stress
python -m benchmarks.spherical_bearings --cache results/displaced-stress/posteriors --summary results/displaced-stress/report.json --output results/bearing-trials/displaced
python -m benchmarks.spherical_bearings --cache results/source-stress/posteriors --summary results/source-stress/report.json --output results/bearing-trials/source
python -m benchmarks.global_posterior --summarize-only --published --trials results/bearing-trials --output results/spherical-trial-comparison
```

The published `posterior.npz` stores the underlying kernel/amplitude draws. The principal `report.json` identifies its field protocol, parent hash, integrated volumes, intervals, prior sensitivities and geographic tests. The baseline volume packet gives parent indices, orientations and mass partitions. Use the preceding report under `bearing/parent-report.json` when reproducing the older continuation. [Implementation](../../src/empirical_isopach/models/mass_mixture/bearings/) · [Earlier release](../../archive/2026-11-20-spherical-continuation/README.md)
