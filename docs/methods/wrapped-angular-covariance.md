# Wrapped angular covariance

The model estimates primary fallout thickness across Earth. Regional thinning can change with distance and bearing. A separately conditioned source-area contribution allows nearby rim sections to differ by orders of magnitude without extending their influence along an unsampled bearing. The same likelihood, priors and source-partition rule apply to all nine eruptions. The maps show one thickness field across all levels. [Matched results and three-scale review](../../comparisons/wrapped-angular-covariance/evidence.md)

## Regional thickness

Work in vent-centred WGS84 Lambert azimuthal equal-area coordinates. Rescale the two chart axes to the authalic disk. Write $r$ for the disk radius in km, $R$ for the authalic Earth radius, $u$ for the bearing, $s$ for a source scale and $q=\sqrt{s^2+r^2}$. The angular taper $h=(r/q)\sqrt{1-(r/2R)^2}$ removes the bearing singularities at the source and antipode.

The conditional mean is an advected, attenuating source kernel. With bearing $b$, reach $L$, eccentricity $e$, $\alpha=1/[L(1-e)]$ and along-fan distance $d=r\cos(u-b)\sqrt{1-(r/2R)^2}$, it is

$$m=\log 100+\log[(1+\alpha q)/(1+\alpha s)]-3\log(q/s)-\alpha(q-s-ed).$$

Mix over six reaches from 30 to 10,000 km, four eccentricities and four core scales. An isotropic candidate carries the same mass as all 16 bearings of any one directional eccentricity at that reach. These scales define priors and do not set an outer deposit boundary.

A Gaussian correction to log thickness uses seven cumulative transitions $g_j(r)=r^2/(r^2+B_j^2)$, with $B_j=3\,4^j$ km and $j=0,\ldots,6$. For each transition the angular covariance has Fourier coefficients proportional to $\exp(-m^2\ell^2/2)$ through order eight. Independent normal sine and cosine coefficients for each order multiply $g_j h^m$. A shared intercept and the radial terms permit azimuthally uniform deposition. The angular length matches the mean-square derivative of a three-harmonic reference. Normalization preserves the reference variance at every radius. Signed cumulative increments can bend or broaden a dispersal fan and permit overlapping lobes. The posterior can also admit a local maximum away from the vent. [Basis and prior weights](../../src/empirical_isopach/models/wrapped_angular/geometry.py)

## What a site measures

The exterior likelihood distinguishes primary thickness from a preserved section. It integrates over independent log scatter and an optional Matérn-3/2 covariance of local bed residuals on spherical chords. These residuals contribute to predicting a new observed bed. The primary isopach integrates the regional coefficients. Measurements and reported minima have the same point likelihood, upper limits use a cumulative probability, and ranges use the probability between both endpoints. No value is assigned to unsampled ground. [Observation policy](reported-minimum-fitting.md)

Where an eruption has an independently mapped source area, its centroid and second moments define an elliptical coordinate system. Inside twice the mapped radius, each regional realization conditions a nonnegative tapered Gaussian excess with a free centre, width and amplitude, a half-probability spike at zero and Student-t section scatter. Interior stations update only this local process. A mapped source without an interior observation retains its source prior. The solver then integrates the joint regional thickness and centroid amplitude to evaluate the marginal thickness quantiles. Sakurajima–Satsuma and custom surveys without a matched source record use the regional field throughout. Equivalent numeric input at the same vent retains its mapped area when a reported minimum is relabelled. [Source records, provenance and uncertainty](../../src/empirical_isopach/assets/wrapped-source-areas.json)

## Inference and volume

The finite mixture starts with 75,264 combinations of regional geometry, covariance and observation scatter. Gaussian conditioning gives the exact point likelihood. A marginal screen retains at least 99.999% of its proposal mass. The code recomputes correlated bounded-event likelihoods for retained candidates, uses two independently scrambled importance populations to correct the bounded Gaussian proposal, and records residuals, effective sample sizes and evidence differences. The fixed three-harmonic sensitivity repeats the fit on the same observations. Its site forecasts are released with each fit. [Regional inference](../../src/empirical_isopach/models/wrapped_angular/regional.py) · [Conditional source fit](../../src/empirical_isopach/models/wrapped_angular/fitting.py)

The principal bulk volume integrates the pointwise median primary thickness over the physical Earth. Integrated pointwise quartiles accompany that estimate. A separate 95% interval integrates coherent fields, each with one set of regional coefficients and its conditionally paired source contribution. The source-area map calculation pairs each stored regional field with 25 local samples. The whole-field interval uses two larger independent populations at 256, 1,024, 4,096 and up to 16,384 draws per population. Each draw receives its own conditional source excess. Both interval endpoints and the median must stabilize under spatial integration, sampling refinement and comparison of the populations. The volume computation uses positive quadrature in disk annuli and an independent Bessel integral for the local excess of each draw. The published map is also integrated on its polar grid and checked with independently scrambled coordinates. The recorded annulus at the farthest station exposes the contribution beyond the survey. [Earth integrals](../../src/empirical_isopach/models/wrapped_angular/integrate.py) · [Alternative distal laws](../../comparisons/wrapped-angular-covariance/thickness-area/README.md)

## Run and inspect a fit

```sh
tephra-isopachs global mazama --geometry wrapped-angular-covariance --output results/mazama-wrapped
tephra-isopachs global observations.csv --geometry wrapped-angular-covariance --output results/custom-wrapped
tephra-isopachs compare mazama --output results/mazama-comparison
```

The family publishes the median map, thickness–area curve, portable joint posterior, fitted and withheld site forecasts, archive scenarios, fold row identities, contour decisions and two whole-Earth integration rules. Geographic blocks, compass sectors and the outer distance quartile are withheld from training. The common contour screen uses replicated coherent fields on each full and geographic deletion fit. Start with the case README and deposit figure. The packet and report retain their input and source identities. [Implementation map](../../src/empirical_isopach/models/wrapped_angular/README.md) · [Replaying the validation](../../benchmarks/wrapped_angular/README.md)
