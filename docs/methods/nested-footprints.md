# Bayesian nested dispersal footprints

The principal nested method jointly fits a declining thickness–area profile and the geographic distribution of that area. Positive angular area increments let successive contours change eccentricity and direction while remaining nested. A smooth, area-preserving angular flow supplies continuously varying dispersal directions. The model uses the same prior for every eruption, without station-centred ridges, invented absences or volume targets. [Matched comparison and three-scale review](../../comparisons/nested-isopachs/adaptive-growth/README.md)

## Why the geography changed

The preceding footprint model allocated a common disk to every growing contour. Its squared radius grew approximately in proportion to contour scale even at large distances. That created an expanding contribution on the source-opposite side, irrespective of the directional evidence. The common disk also imposed nearly circular proximal contours; ellipse eccentricity and displacement were coupled. Turning components moved their centres but retained fixed ellipse axes. These restrictions could obscure proximal asymmetry and changes in dispersal direction.

The earlier station-corridor model had another limitation: each positive station generated a full-strength segment to the source. Moving a survey line could move a modeled ridge. None of the new shapes is constructed from station rays. Locations enter only through the observational likelihood. The original whole-area sensitivity remains in `nested-whole-area/`; earlier corridors and footprints retain their identities and source bytes. [Corridor rules](nested-reach.md) · [Whole-area model](nested-isopachs.md) · [Preceding footprint protocol](https://github.com/siyulai21/tephra-fallout-isopachs/blob/514572583423c861d5cb5a43e4e6bbd963f42cf8/docs/methods/nested-footprints.md)

Elliptical increments are empirical descriptions of preferential dispersal, not a reconstruction of atmospheric transport or an identification of eruption phases. Multiple phases, topography, aggregation and preservation can produce departures from this representation. No lack of sampling is interpreted as non-deposition. [Sulpizio et al. (2024)](https://doi.org/10.1007/s00445-024-01753-5) · [Yang and Jenkins (2023)](https://doi.org/10.1007/s00445-023-01652-1)

## Positive area growth

Work in the source-centred WGS84 equal-area chart. Let $K(\theta;a,f)$ be the squared ray radius of an ellipse with semiaxes $a,1/a$ and centre $(af,0)$, where $0\leq f<1$. Its mean over bearing is one. Aspect and source displacement are independent, so an elongated contour need not have a predetermined upwind reach.

At scale $t$ and common transition length $c$, split squared radius into

$$
C(t)=\frac{c^2t^2}{c^2+t^2},\qquad D(t)=t^2-C(t).
$$

Both are increasing. The common proximal component approaches a finite area instead of expanding indefinitely. Its kernel is $K_0(\theta)=K(\theta;\sqrt a,0.35f)$, so proximal symmetry is permitted but not imposed. For an increment at scale $u$, define

$$
a(u)=\sqrt a+(a-\sqrt a)\frac{u}{u+c},\quad
f(u)=f\left(0.35+0.65\frac{u}{u+c}\right),\quad
\phi(u)=\phi_\infty\frac{u}{u+4c}.
$$

Before the additional angular flow, the primary contour has squared radius

$$
R^2(t,\theta)=C(t)K_0(\theta)+
\int_0^t K\big(\theta-\phi(u);a(u),f(u)\big)\,dD(u).
$$

Each sector receives positive area increments even when their preferred direction changes. Therefore $R(t,\theta)$ increases with $t$. All levels belong to one nested field; their shapes are not separately fitted or subsequently smoothed. Before physical-domain clipping, the primary area is exactly $\pi t^2$.

For a two-phase shape, add $h(t)^2K(\theta-\psi;a,f)$, with

$$
g(t)=\sqrt{D(t)},\qquad
h(t)=\eta g(t)+(0.8-\eta)\frac{4c\,g(t)}{4c+g(t)}.
$$

The secondary increment can approach a finite reach ($\eta=0$) or remain persistent ($\eta=0.3,0.8$). Its positive area contribution preserves nesting through overlapping lobes. This is a model of combined isopachs, not separate phase volumes. The detached alternative retains a geometric union with a growing remote patch, including draws whose maximum lies away from the source. Clipping and union areas include all patches. [Area-growth implementation](../../src/empirical_isopach/models/nested/footprints/growth.py)

## Continuous directions without changing area

Apply the same smooth coordinate transformation to every level in a draw:

$$
F(r,\theta)=\left(r,\theta+\delta+b\frac{r}{r+4c}\right).
$$

Here $\delta$ completes a continuous uniform bearing prior within a catalogue orientation bin, and $b=120^\circ\tanh(z_b/2)$ with $z_b\sim N(0,1)$. Evaluate the field using $F^{-1}$. The determinant of this transformation in equal-area polar coordinates $(r^2/2,\theta)$ is one. It preserves the physical disk, every contour area, containment and topology. It can represent a smoothly bending lobe without requiring its final outline to be star-convex. This transformation is part of the fitted geography, not a modification of plotted contours. [Implementation and inverse](../../src/empirical_isopach/models/nested/footprints/flow.py)

The family prior remains 0.20 round, 0.40 single, 0.15 turning, 0.20 two-phase and 0.05 detached. Each family divides its mass evenly over its catalogue and bearings. The common transition scales are 2, 8, 32, 128 and 512 km. Single shapes use $a=1,1.8,3$ and $f=0.2,0.55,0.85,0.97$; turning shapes use $a=1.4,2.4$, $f=0.55,0.92$ and $\phi_\infty=\pm50^\circ,\pm100^\circ$. Two-phase shapes use $a=1.8$, $f=0.65,0.92$, $\psi=\pm75^\circ,\pm145^\circ$ and the three secondary persistence values above. Detached offsets remain 50, 250 and 1,250 km. Forty-eight orientation bins plus uniform within-bin displacement cover the full circle continuously.

This prior allows concentrated and broad fallout. Weakly observed bearings remain conditional extrapolation; a narrower posterior lobe does not establish absence outside it. Every release reports sensitivity to family odds, transition scales, secondary persistence, bending and outer transfer, with effective sample sizes for importance reweighting. Sparse data may still favor a rounded deposit or leave several directions plausible.

## Thickness, bounds and volume

For a point, find the first nested region containing its inverse-transformed position. Its whole-area rank is $q=\sqrt{A/\pi}$, where $A$ is the complete region's area clipped to the physical Earth. The draw assigns thickness $T(q)$. Its bulk volume is

$$
V=2\pi\,10^{-5}\int_0^{2R_\mathrm{authalic}}q\,T(q)\,dq
$$

for kilometres and centimetres. Area redistribution at fixed thinning preserves this integral. Fitting a different geography changes the ranks of observations and can therefore change the inferred thinning and volume. The existing profile, robust observation distribution, exponential/power tail mixture and independently derived outer transfer remain explicit. The continuation joins beyond all training ranks, so it cannot change a fitted observation. [Thinning and transfer equations](nested-isopachs.md)

Each reported lower-only thickness enters as a measurement at its stated minimum, at every distance. Its qualifier has no numerical effect. Upper limits contribute cumulative probabilities; two-sided ranges contribute the probability between both endpoints. The fused likelihood retains the 95% ordinary and 5% broad normal components in log thickness. It forms interval widths before subtracting the predicted mean and uses stable tail probabilities, including narrow intervals. There are no added observations, detection limits or error estimates. [Observation policy](reported-minimum-fitting.md)

The principal estimate integrates the **pointwise posterior median thickness over Earth**. The conditional band separately integrates pointwise 25th and 75th thickness percentiles on the same domain. Complete-draw volumes serve as sampler and independent-integration checks; their quantiles are not substituted for the thickness-map band. Sector tables and fractions inside the first measured rank and outside the last positive rank identify extrapolation. The shared support audit retains holes, rejected contours and every rejection reason. [Support protocol](supported-isopachs.md)

## Sampling, replay and limitations

Four independent populations fit both distal laws using tempered sequential Monte Carlo. Thinning and direction receive separate symmetric Metropolis updates, with proposal scales adapted between tempering stages. Reversible geometry proposals jointly transport profile scale and height; their acceptance uses the exact prior and likelihood ratios. All likelihood calculations use only the current training rows. Held-out observations do not select geometry or sampling settings.

Convergence requires agreement between populations on intrinsic and continued volume quantiles **and** median predicted log thickness at training sites. The median and 95th percentile of the between-population site ranges must be below 0.15 and 0.40 natural-log units. Full fits also check the mapped geography on 1,536 physical points, including unsampled bearings. Volume criteria retain the preceding median ratio of 1.35 and tail ratio of 2. Failed production fits receive the same refinement policy across eruptions: 768 particles per population, doubling through 6,144, then separate 12,288- and 24,576-particle attempts when required. The fixed auxiliary comparison uses a 12,288-particle ceiling for both nested methods. Checkpoints save each completed population attempt, its posterior hash, input, seed and engine identity; verified interrupted runs resume at the next particle count.

If the larger production attempt still fails, a fixed-target Metropolis stage evolves 3,072 particles from each of the four populations. A fixed pilot proposal draws the geometry and thinning law together with all eight continuous coordinates. Conditional multivariate Student-t proposals have five degrees of freedom; their complete discrete and continuous proposal densities enter the Hastings ratio. The pilot changes only proposal efficiency. Local geometry, profile and angular moves retain access to states absent from that pilot. An additional thinning-law flip transports the peak thickness with an exactly reversible, unit-Jacobian amplitude shift. Neither move changes the likelihood, geological priors or continuation law.

The populations retain separate states and random moves, conditional on the common fixed proposal; they are never pooled and resampled to manufacture agreement. Local proposal covariance adapts only during the first 256 moves. Checkpoints at 256, 512, 1,024, 2,048 and 4,096 moves retain all states and diagnostics. Selection requires at least 1,024 moves, two successive passing population checks, and less than 10% change in each pooled complete-volume quantile since the previous checkpoint. The same rules apply to every unsuccessful production partition. The separately signed sampler source and pilot posterior are part of the fit identity. Unresolved warnings remain visible at the ceiling. [Implementation](../../src/empirical_isopach/models/nested/footprints/rejuvenation.py)

Area increments use midpoint integration in area on a fixed logarithmic scale grid, with angular normalization independent of observations. Grid-refinement tests check ranks, nesting and clipping. Two independently scrambled geographic quadratures compare coherent draw volumes against the profile integral. Posterior draw count, mapped geometry and physical-grid resolution receive separate refinement checks. The draw check evaluates the 25th, 50th and 75th thickness percentiles on two independent geographic grids. It requires at least 2,048 draws, less than 1.5% change in every integrated quantile, and a 95th-percentile visible-field change below 0.035 log10 cm for every surface. It doubles the display sample up to 16,384 draws or the available posterior, retaining a warning if these criteria fail. Finite posterior quantiles can retain small sampling fluctuations; no contour smoothing is applied to hide them.

Use a fresh destination for fitting:

~~~sh
tephra-isopachs global Mazama --geometry nested-isopachs --output results/mazama-footprints
~~~

The comparison includes four buffered geographic refits, quadrant omissions, known deposits, actual withheld campaigns, source-bound checks and proximal/intermediate/full-scale plates for every registered eruption. These eruptions and several known-deposit scenarios informed development. Better development scores alone do not establish universal predictive calibration. Unsurveyed lobes, unobserved proximal thickness, preservation and missed phases remain structural uncertainties even when numerical checks pass. [Reproduction and released evidence](../../comparisons/nested-isopachs/adaptive-growth/README.md)
