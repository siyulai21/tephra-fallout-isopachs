# Continuing a measured fallout fan across a gap

The three global families retain their fitted observation models. Each uses the locations of positive training records to control how its thickness field continues into unsampled ground. Every exact reading, lower bound and upper bound keeps its original likelihood. Every retained measurement triangle keeps its field. Continuation therefore leaves the fitted posterior parameters valid and produces a different distribution of complete deposits. [Catalogue results and controlled challenges](../../comparisons/global-extent/README.md)

## Sampling geometry

Calculations use the source-centred WGS84 equal-area chart. Positive locations define finite segments from the vent. A purely upper bound protects its own location. Duplicate locations count once in the geometry. Let $s$ be the median distance of the three nearest distinct positive stations, using all available when fewer than three exist. For along-track coordinate $a$, cross-track offset $b$, and station range $R_i$, write $u=\mathrm{clip}(a,0,R_i)$. The corrected squared transverse displacement in units of spread is

$$
C_i=\frac{b^2+\min(a,0)^2}{s^2+2su}.
$$

The width grows as the square root of distance in the far field. This is a structural alternative to a fan whose width grows in direct proportion to range. A smooth transition with width equal to the median nearest-neighbour separation of exact sites protects the measured footprint and individual stations, including bounds. The surveyed axial concentration is $\left|\sum R_i^2 e^{2i\theta_i}/\sum R_i^2\right|$. This survey statistic gives a long, narrow transect less authority over its crosswind extent. Past the farthest positive range $R$, its influence decays as $L=1/[1+(\max(r-R,0)/R)^2]$. These assumptions are shared by all eruptions. An unsampled sector can still contain fallout.

## Spherical mixture

Displaced circular kernels can fit stations on a flank and imply an enormous unobserved peak. For each kernel in each draw, find its highest log density at any positive station or the vent. Monotonicity about the kernel centre makes this an exact nearest-site calculation on the sphere. If the proposed log density rises by $x>0$ above that value outside the protected geometry, allow a further rise of $b\tanh(x/b)$. Two finite-spread alternatives use $b=\log 4$ and $\log 2$. At the join both value and slope match. This limits an unsupported rise in an individual component while allowing genuine detached or off-vent maxima.

Calculate the distance to the nearest finite segment with $C_i$ and a downstream endpoint term scaled by $4(s^2+2sR_i)$. Evaluate this directional evidence at the last surveyed radius for farther targets. A positive polynomial taper with half thickness at one scaled displacement has width multipliers 1.5 and 0.75. The broad alternative applies half the log taper. Each retains a diffuse skirt, initially one half or one quarter, decreasing by up to three quarters according to axial concentration and $L$. The taper strength also accounts for the unsampled fraction of the component peak and diminishes beyond the surveyed range. The full rule and constants are public. The existing source-focused B-Tm kernels already put their maximum at the vent and retain their original field exactly. [Spherical continuation](../../src/empirical_isopach/models/mass_mixture/bearings/flank.py) · [Underlying spherical likelihood and mass normalization](probabilistic-global.md)

## Nested corridors

A site can establish a long axis. In the earlier finite laws the maximum permitted crosswind width grew in proportion to distance on that axis, even for a single distal observation. For each of the nine combinations of width and downstream reach, the revised model uses $C_i$ for transverse spread and preserves its longitudinal thinning. It blends in a positive diffuse skirt. It also evaluates the best intrinsic area rank on sampled bearings at each radius, gradually discounting bearings beyond their last positive site. A fitted turning ridge that thickens above this sampled flank receives the same smooth cap with allowances of a factor of four or two. The broad alternative takes half the crosswind correction. The original whole-area structural law in the fitted posterior persists in both alternatives. Its mass can dominate the upper interval.

The model can still miss a fan present in an unsurveyed sector. Its earlier linear-spread result, four independent survey comparisons and tests with a deliberately hidden fan are published beside the revised estimate. [Nested continuation](../../src/empirical_isopach/models/nested/spreading.py) · [Parent corridor laws](nested-reach.md)

## Directional ensemble

The family still contains six fan structures and a coherent draw of uncertain fan azimuth. Its compact angular basis can give each isolated observation its own preferred range. Before predicting between sites, convolve the periodic range basis with a Gaussian of angular standard deviation 11.25° or 22.5°. Smoothing fades to zero in the retained triangles and at all training sites. The release from an isolated site also accounts for its distance from the prediction point relative to local spacing plus $r\sin(22.5°)$. Far beyond the survey, $L$ preserves uncertainty in narrow, long-distance transport. The change can increase or decrease modeled thickness. [Angular basis and release](../../src/empirical_isopach/models/directional/azimuth/resolution.py) · [Uncertain fan azimuth](uncertain-azimuth.md)

## Prediction, volume and uncertainty

The two finite alternatives enter each family as a mixture of **complete deposits**. Geographic scores from other eruptions determine their weights. Each eligible donor has the weight of four geographic blocks, divided evenly among its passing blocks and then among sites in each block. A symmetric Dirichlet(2,2) regularizer gives finite weights. The target eruption and its input hash are excluded. Failed donor fits remain in the reporting ledger. The original family continuation is published as a wider sensitivity. [Scoring and fold identities](../../src/empirical_isopach/models/extent_validation.py) · [Transfer weights](../../src/empirical_isopach/models/extent_weights.py)

The central estimate integrates the pointwise median over the physical Earth. The 95% interval uses the 2.5th and 97.5th percentiles of coherent whole-deposit volumes. Two independent scrambled quadratures and increasing node counts check the integrals. Splitting coherent draws checks sampling variability in the interval endpoints. The draw check refines the spherical and nested samples up to their stated ceilings. Spherical results also have an independent check using samples from the normalized kernels. All eight fixed spherical prior fits keep separate intervals and numerical flags. The map grid, prior-sensitivity envelope, support screen and reported posterior interval answer different questions.

These rules were developed after inspecting the named anomalies. Leaving a target out of the weight calculation protects that calculation. The nine-eruption comparison is a development evaluation. It supplies no independent measurement of global volume.
