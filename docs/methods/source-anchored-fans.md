# Source-focused fallout fans

[Project home](../../README.md) · [Preceding global model](probabilistic-global.md)

This revision investigates an unsupported thickness maximum in the B-Tm reconstruction. The preceding model represented fallout as circular patches with freely displaced centres. A positive, finite integral did not stop those patches from placing the thickest deposit away from the known volcano or spreading mass across unobserved sectors.

On a 5 km grid within 400 km of the B-Tm vent, the preceding posterior median surface peaked approximately 94 km southwest of the vent, at 163 cm. Its vent prediction was 73 cm, whereas the closest accepted observation was 150 cm at 2.6 km distance. The vent coordinates in the input were correct. This is a model-extrapolation problem, not a coordinate correction. The supplied centimetre convention is unchanged; the original B-Tm and Aso-4 collections still require source-level audits before claiming every record has been independently verified.

## Shape constraint

The candidate anchors each fan at the supplied vent. Its thickness is positive and nonincreasing along every ray from that vent. Several fans can overlap, with different scales and directions. Their sum has its maximum at the vent and every nonempty positive superlevel set contains the vent. This is a declared general shape assumption, not a synthetic measurement of vent thickness.

Real deposits can have secondary maxima: [Watt et al. (2015)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JB012116) document one in the Mount St Helens deposit. Co-ignimbrite fallout can also have a distributed effective source. The constraint therefore needs predictive checks and a comparison with less restrictive geometry. It must not be presented as a universal law that every observed deposit is thickest exactly at a point vent.

## Source-focused ellipses

Use the same WGS84 authalic equal-area sphere as the preceding model. Let `r` be chord distance from the vent (the equal-area polar radius), `theta` its azimuth, and `0 <= r <= 2R`. Surface area is exactly `r dr dtheta`.

For each fan, an unconstrained vector `b=(b_x,b_y)` determines direction `phi` and elongation `delta=|b|`. Set `C=cosh(delta)` and `S=sinh(delta)`. Define

```text
L(theta) = L / [sqrt(C) (C - S cos(theta-phi))].
```

Each constant `r/L(theta)` contour is an ellipse with one focus at the vent in the equal-area chart. `L` is the planar area-equivalent scale: the area enclosed by `r=L(theta)` is exactly `pi L^2`, regardless of elongation. The upwind/downwind extent ratio and crosswind width are coupled by ellipse geometry. There is no independently wandering patch centre or independently adjustable unobserved lobe.

Each fan mixes the same three thinning profiles as before:

```text
Gaussian:     exp[-r^2 / (2 L(theta)^2)]
Exponential:  exp[-r / L(theta)]
Power:        [1 + r^2 / (2 q L(theta)^2)]^(-q).
```

Every radial integral over the full Earth is analytical. The normalizer is its remaining one-dimensional periodic angular integral. No map boundary, closing isopach, missing-data zero, or analyst-selected distal radius enters that integral.

This directional construction is a field on the source-centred equal-area chart. At the antipode all azimuths meet, so anisotropic profiles can have direction-dependent limiting values there. The assigned value at that single point has zero effect on the area integral, but the model is not a globally smooth transport solution. Diffuse-tail interpretations remain especially dependent on this empirical geometry; finite-Earth normalization alone does not validate intercontinental dispersal.

For narrow fans, uniform azimuth quadrature is inefficient. An eccentric-anomaly change of variable resolves this without moving a numerical grid by hand. With `e=tanh(delta)`,

```text
cos(theta-phi) = (e+cos E)/(1+e cos E)
L(theta(E))   = L sqrt(C) (1+e cos E)
dtheta/dE     = 1 / [C (1+e cos E)].
```

The planar angular area weight simplifies to `L^2 (1+e cos E)`. The finite-Earth correction starts with 256 periodic nodes in `E`. Every saved posterior field is checked by doubling and quadrupling the angular resolution. If any component changes by more than `10^-6`, the entire posterior is refitted at twice the resolution, up to three doublings. Earlier resolutions remain recorded; draws are never corrected after fitting. This is a reproducible numerical refinement rule, not a change to physical priors or a geological confidence threshold. Refinement agreement is a numerical error estimate, not an analytical error bound.

Normalizing each fan and mixing it with nonnegative mass weights keeps total volume as the fitted volume parameter. The pointwise median map is still not integrated to obtain the median volume. The likelihood retains exact observations, upper/lower bounds, known errors and the broad-error sensitivity framework; residual scatter does not add physical deposit mass.

## Development evidence and comparisons

The first attempted correction used a four-coefficient harmonic directional scale with an anchored maximum. On the same accepted observations, it reduced B-Tm's median from 344 to 143 km³ but increased Aso-4's median from 1,836 to 3,354 km³. Both samplers passed their numerical checks. Anchoring alone therefore did not solve the volume problem. Those outcomes are retained rather than selected away; they motivated testing a more coherent downwind shape.

Published eruption volumes are context, not fitting targets. [Takarada and Hoshizumi (2020)](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.00170/full) give 590–920 km³ of bulk Aso-4 fallout, whereas [Rougier et al. (2022)](https://doi.org/10.1093/gji/ggac131) give 220–370 km³ using their statistical and expert-boundary procedure. Their disagreement is itself a warning against treating one target range as observed truth. [Costa et al. (2024)](https://www.nature.com/articles/s43247-023-01162-0) report Paektu fallout in DRE and model its phases with atmospheric transport; comparisons require keeping bulk fallout, DRE and PDC products distinct. None of those published totals is inserted into this model's likelihood or priors.

Model adoption depends on source geometry, numerical checks, matched geographic predictions, known-truth failures and sensitivity to explicit assumptions. A lower volume by itself is not sufficient evidence of improvement.

## A less restrictive experimental field

A second construction allows smaller secondary maxima while retaining a maximum at the volcano. On the authalic sphere, let `g0` be a source-centred Gaussian/von Mises–Fisher profile with peak one, and let `h = sum(wk gk)` be a convex combination of Gaussian profiles, each with peak one. With `0 < A < 1`, define

```text
f(s) = g0(s) + [1-g0(s)] A h(s)
T(s) = 10^5 V f(s) / integral_Earth f(s) dA.
```

At the source, `f=1`. Elsewhere, `f <= g0+(1-g0)A < 1`. Thus the source is the unique global maximum, while the displaced profiles can create lower secondary peaks. This is a mathematical constraint, not a synthetic observation. The model cannot represent a genuinely higher distal maximum.

Products of two Gaussian/vMF profiles are proportional to another vMF profile. The whole-Earth normalizer is therefore analytical: `Z = Z0 + A sum(wk [Zk-Z0k])`, including every overlap term. Gaussian limits and coincident/opposite centres are tested; an independent whole-Earth area quadrature verifies the resulting mass. Partition calculations use positive quadrature of `g0` and `(1-g0)gk`; their mass residual and refinement error are reported separately and never determine the total volume.

The experimental measured-location variant fixes secondary centres using training coordinates only. Repeated coordinates are removed, coordinates are lexicographically ordered, the first centre is the site nearest their arithmetic mean on the sphere, and subsequent centres maximize distance to the selected set. Each reference width is the nearest selected-centre distance. A common multiplier has `log(width multiplier) ~ Normal(log(0.5),0.7)`. Each fold rebuilds this design from its training sites, without heldout locations or thicknesses. The 6- and 12-basis development trials both remain visible. Their narrow conditional intervals do not incorporate the possibility of a missing, unobserved fan.

Unlike the source-focused three-profile mixture, this envelope has **Gaussian tails only**. Smaller volumes or narrower intervals can follow from that stronger restriction. Neither the local basis design nor its finite total proves that missing deposit mass is small. The exposed `source-envelope` protocol is experimental and includes a heavy-tail source-fan comparison; it is not automatically promoted when it returns a lower volume.

Fixing centres at measured sites restricts the basis functions, not the locations of every secondary maximum. Overlap can create peaks between sites, including in unsampled gaps. Such peaks still need evidence checks; they are not made observationally supported by the basis design.

## Reproduction and decision scope

`tephra-isopachs global B-Tm --geometry source-fans --output results/btm-source` runs the source-focused family with all seven fixed sensitivity refits and four matched buffered geographic holdouts. `--geometry displaced` reproduces the preceding scientific specification in a fresh cache. `--geometry source-envelope` runs the experimental bounded field and its declared alternatives. A catalogue entry can record a reviewed geometry choice; it never supplies eruption-specific wind, particle or volume parameters.

Geographic validation can reveal poor observation prediction; it cannot identify the unseen total mass. Choosing a family after inspecting its holdout scores also makes those scores development evidence, not an unbiased test of the selection procedure. The repository therefore reports the competing scores and failed alternatives explicitly and does not claim a universally superior global estimator.
