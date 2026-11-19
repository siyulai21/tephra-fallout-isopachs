# Whole-Earth continuation from supported isopach fragments

This is the underlying directional field and the original single-structure study. The active estimate now combines six field structures. [Median isopachs, weights and volume definitions](directional-ensemble.md)

The input is the accepted fallout thickness table and its vent. The model fits all accepted fans jointly, interpolates latent thickness within locally sampled triangles, and completes the field with a directional thinning law. Every eruption gets the same priors, geometry rules, sensitivities and geographic tests. The integral is over the WGS84 surface. Results are in km³ of bulk fallout. [Cross-eruption results](../../comparisons/directional/single-structure/README.md) · [Implementation](../../src/empirical_isopach/models/directional/)

## What a measured contour contributes

Distinct exact locations define the [existing support mesh](supported-isopachs.md). Its third-neighbour spacing and triangle shape tests leave holes and disconnected patches. Each retained vertex has a latent log thickness $z_i$. Within a triangle with barycentric coordinates $w_i$, log thickness is $\sum_i w_i z_i$. Every level between vertex values is a straight segment. Every draw integrates the full continuum of these levels. Drawing a denser set of isopachs has exactly zero effect on the inferred volume.

For visualization, 25 log levels span the 5th through 95th percentiles of exact measurements. A segment is drawn on the evidence panel when exact measurements at distinct vertices bracket the level and at least 90% of posterior fields cross that level in the triangle. The segment is drawn on the mean-log node field. The probability concerns a crossing somewhere in that triangle. Its location varies across draws. The 90% display threshold has no role in fitting or integration. The complete triangle-by-level table includes unbracketed and uncertain levels. The shared complete-contour screen is applied independently to the directional median surface and the GP and spline surfaces. It supplies separate evidence about closure and deletion stability. [Engwell and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC4498447/)

Outside the footprint a transition of one median nearest-neighbour spacing joins the interpolation to the global field $B$. A bevelled polygon buffer and constrained triangulation define this strip. The buffer is limited at the physical edge of the equal-area chart. Each boundary vertex keeps its original latent thickness. At new outer vertices the local weight is zero. For points in the strip,

$$
\log T(x)=\sum_i w_i(x) z_i+
        \left[1-\sum_i w_i(x)\right]\log B(x).
$$

Here $w_i$ is the barycentric weight of a measured mesh vertex. The outer mesh nodes specify where local interpolation ends. They supply no observations or likelihood terms. Every point in the strip counts as extrapolation, including points in small holes. Beyond it $T=B$. As a result, distant gaps cannot acquire a new peak from continuing a local spline polynomial. Actual secondary maxima within measured ground remain possible. Secondary maxima have been documented in fallout deposits. [Eychenne and colleagues](https://doi.org/10.1002/2017JB014412)

## How far a fan can extend

In the source-centred authalic equal-area chart, $r$ is chord distance in km, $\theta$ is clockwise bearing from north, and $dA=r\,dr\,d\theta$ for $0\le r\le 2R$. The background is positive and thins along each ray. It has a compact proximal contribution and a directional contribution with a separately fitted share of the vent thickness,

$$
 B=T_0\{(1-m)F(r,h_0,\xi)+mF(r,h(\theta),\xi)\},\qquad
 F=(1+\xi r/h)^{-1/\xi}.
$$

The $\xi=0$ limit is exponential. The remaining profiles have distal power exponent $1/\xi$. A common slope at the origin permits comparison at a fixed proximal scale $h$. A normalized posterior field is $T(x)/\int T\,dA$, and its associated total is $10^{-5}\int T\,dA$. We fit thickness directly, so the prior on total volume is induced by the geometry and thickness priors. Each sample has its own integral.

To choose directions, sort distinct exact sites by decreasing radius. Take the first bearing and add another if it is separated from every selected bearing by at least half the smaller of their two angular half-widths. Coordinate order resolves radius ties. A site belonging to a retained triangle has a 45° half-width. An isolated site has 22.5°. The reference radius of an accepted direction is the actual radius $d_j$ of that site. At zero radius a site has no bearing. These rules operate anew within each training fold.

Use $K_j(\theta)=(1+\cos(\pi\Delta_j/a_j))/2$ within half-width $a_j$, zero elsewhere, where $\Delta_j$ is circular angular distance. Set $w_j=K_j/\max(1,\sum_k K_k)$ and
$h(\theta)=h_0+\sum_j w_j d_j\exp(b_j)$. Normalizing overlapping kernels prevents a cluster from multiplying the directional length just by contributing more bearings. A short proximal reach gives a smaller prior extension than a distal reach of the same bearing. A measurement on one bearing can still influence nearby bearings and the shared proximal scale.

Where all angular weights vanish, both contributions use $h_0$. This is a **design-dependent assumption about continuation in gaps**. Exposures and cruise locations depend on access and preservation. Sparse sampling alone gives no geological evidence that a fan was absent. One sensitivity adds a fan at the midpoint of the widest gap between bearings of the outer half of distinct exact sites. It uses the same multiplier prior, a 45° half-width, and the 75th percentile of all exact radii as its reference reach. The resulting change measures dependence on one plausible missing fan. More, broader, or differently aligned hidden fans remain possible.

## Priors, likelihood and uncertainty

Natural logarithms are used throughout. Independent priors are common to every deposit.

| Quantity | Prior or rule |
|---|---|
| Background vent thickness $T_0$, cm | $\log T_0\sim N(\log 100,2.5^2)$ |
| Common proximal scale $h_0$, km | $\log h_0\sim N(\log 35,1.4^2)$ |
| Directional reach multipliers | $b_j\sim N(-\log 3,1^2)$ |
| Vent thickness share of the directional contribution | $\operatorname{logit}(m)\sim N(-1,1.5^2)$ |
| Distal shape | $\xi\sim\operatorname{Beta}(0.5,1.5)$, shared across directions |
| Log observation scatter | $\log\sigma\sim N(\log 0.7,0.8^2)$ |
| Local node departures | $z_i\sim N(\log B(x_i),\tau^2)$, conditionally independent, reference $\tau=\sigma$ |

The prior on $\xi$ gives weight to nearly exponential thinning and to slow distal thinning. Its interval is an empirical choice. On a finite Earth even $1/\xi\le2$ has finite mass. No grid window, detection threshold or hand-drawn closing ring truncates that tail. At the antipode, bearing has multiple limits. The single point has zero area.

Exact measurements enter once. Colocated exact measurements share the same node. For a reported log error $s_i$, the observation variance at location $x$ is
$$
  \sigma^2+s_i^2+\tau^2(1-\sum_j w_j(x)^2).
$$
The last term restores unresolved site variability as local interpolation weights diminish. Combined with the independent prior node effects it gives the same marginal local variance at a vertex, between vertices, and in a gap. This is an observation-scale assumption. Site scatter never adds deposit volume. Lower and upper limits contribute the corresponding Gaussian tail probabilities. The likelihood can include a bound within a retained triangle even though a bound supplies no new mesh vertex. Legacy pseudo-rows are ignored by the shared reader.

Exact measurements at nodes are analytically marginalized. Node innovations involved in overlapping measurements or limits are jointly sampled when needed. The other nodes are drawn from their exact conditional Gaussian distributions. Four NUTS chains retain complete fields. A numerical retry doubles warmup and sampling with a new seed and tighter target acceptance. Parameter and integrated-volume diagnostics report rank-normalized split R-hat, bulk and tail ESS, divergences and energy diagnostics. The acceptance criteria are R-hat <1.01, ESS ≥400, no divergences, and minimum energy BFMI ≥0.3 for the sampler. A failed final run remains in the comparison with its status.

The fixed sensitivities include exponential, eighth-, fourth- and second-power tails, half and double angular widths, an additional missing fan, strict and permissive support, a threefold broader median proximal-scale prior, twice the transition width, and twice the local-to-observation scatter ratio. Each changes one policy from the reference. Their spread describes structural sensitivity. Preservation, correlation errors in deposit attribution and an entirely unseen fan can lie outside any reported interval. Literature volumes and eruption-specific winds do not enter the calculation. [Yang and Jenkins](https://doi.org/10.1007/s00445-023-01652-1)

## An unobserved proximal cap

A triangle can span the vent despite a gap around the source. Then the fitted local source thickness can differ markedly from the background value $T_0$. We evaluate a common positive proximal addition as a separate sensitivity. For the distance $a$ to the nearest accepted observation or bound and the complete reference field $T$, use

$$
 T_{\mathrm{cap}}(x)=T(x)+D\,[1-(r/a)^2]_+^3,\qquad D=[T_0-T(0)]_+.
$$

A site at the vent makes the added volume zero. Otherwise the added bulk volume is exactly $10^{-5}\pi a^2D/4$ km³. The added field has two continuous derivatives at the edge and vanishes at every accepted row. Both fields therefore have the identical likelihood and the same posterior for the original parameters. Each draw gives its own cap-inclusive volume. Geographic and quadrant holdouts rebuild the cap from the training rows and rescore newly affected test sites. The amplitude and taper define one possible unrecorded proximal deposit. Neither supplies an observational upper bound. The report also compares finite searches for the thickest point with and without this cap, including actual sites and a finer near-source grid.

## Mass accounting and verification

Each ray is split at its intersections with every retained and transition triangle. Gaps integrate the radial thinning law analytically, including its exponential and second-power limits. Inside triangles, positive Gauss weights integrate the actual latent field. Angular panels split at mesh vertices, fan edges and compass boundaries. Refinement must change the sum of absolute differences across the entire five-region by eight-bearing mass budget by less than 0.5% of total for **every** reported field. Integrated footprint and transition areas must agree with their independent polygon areas to 0.1%. Refinement agreement estimates numerical error. Cartesian area integration and independent radial quadrature are also tested.

The five disjoint regions are retained triangles, transition, remaining proximal area inside the nearest exact radius, remaining gaps between the extreme exact radii, and remaining area beyond the farthest exact site. They exhaust the physical Earth in every draw. Compass-sector quantiles and component quantiles need not add to total-volume quantiles. Whole-field chain/draw arrays allow any group of sectors to be summed before taking its interval.

If an unsampled open region can contain a smooth, nonnegative lobe, scaling that lobe changes integrated mass without altering any measured value or bound elsewhere. Thus the observations alone provide no finite upper bound in this more general field class. Our conditional upper limits come from the stated thinning, fan and scatter assumptions. The paired-truth test quantifies one finite example.

The validation record contains the same four buffered geographic folds as the regional peers and spherical mixture, plus deletion of each 90° quadrant using the same buffer. Empty or unestimable quadrant tests stay visible. All training geometry is rebuilt without withheld points. The comparison also includes known-volume deposits and two distinct truths that give *identical observations* in a sector-gap experiment.

Run the common directional calculation on its own with the following command. The full upload workflow also runs it automatically.

~~~sh
tephra-isopachs global Aso-4 --geometry directional --output results/new-directional
~~~

[Case results and stress tests](../../comparisons/directional/single-structure/README.md) · [Other global estimate](probabilistic-global.md)
