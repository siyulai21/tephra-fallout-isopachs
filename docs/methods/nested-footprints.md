# Bayesian nested dispersal footprints

The model fits a declining thickness profile against the square root of deposit area. Each draw explains where that area lies using a round, elongated, turning, two-phase or detached footprint. All eruptions share the same geographic prior. Thicknesses, bounds and locations determine the posterior. The Minoan Plinian and co-ignimbrite observations enter a single combined deposit. [Catalogue and paired tests](../../comparisons/nested-isopachs/geography/README.md)

## Why the geography changed

In the corridor model each positive station generated a full-strength segment between itself and the source, even for a very thin bed. The strongest of the overlapping segments determined the thickness multiplier. Moving a survey line could thus move a ridge between otherwise identical field draws. The finite-spread revision narrowed individual corridors. The original whole-area alternative avoided this sampling imprint, but taking the larger of two directional reaches could still produce a sharp concave junction. A trial with two equally persistent moving footprints also failed at Mazama, where a distal extension of the secondary lobe followed from its assigned reach. [Corridor rules](nested-reach.md) · [Original whole-area geometry](nested-isopachs.md) · [Archived trials](../../archive/2026-11-23-nested-geography/README.md)

An ellipse is an empirical representation of dispersal with a preferred direction. The circular proximal limit, scale-dependent migration, allocation between phases and discrete priors below are modeling choices introduced here. An eruption with strong proximal asymmetry, more phases or a long unseen lobe can violate those choices. Iso-area surfaces give a practical way to distinguish assumptions about the shape of an isopach from assumptions about thinning. [Sulpizio et al. (2024)](https://doi.org/10.1007/s00445-024-01753-5) · [Yang and Jenkins (2023)](https://doi.org/10.1007/s00445-023-01652-1)

## A nested deposit from every draw

Use the source-centred WGS84 equal-area chart. At positive scale $t$, let $s$ be the migration length and $g=t^2/(t+s)$. A common disk contributes squared radius $t^2-g^2$. The primary dispersal ellipse has semiaxes $ag,g/a$ with $a=(1-e^2)^{-1/4}$. In the unit-aspect coordinates of the ellipse its centre, before rotation by the main bearing, is

$$
c(g)=e\left[\frac{g^2}{g+s}(1,0)+\frac{g^2}{g+4s}\big((\cos\phi,\sin\phi)-(1,0)\big)\right].
$$

The derivative of $g^2/(g+s)$ lies between zero and one and decreases as $s$ increases. The centre's velocity is therefore a weighted sum of the initial and final unit directions with total coefficient below one. Each ellipse strictly contains its smaller predecessors. It contains the source. Its area is $\pi g^2$. The shared disk dominates at small $t$, where the ratio of radii in any two directions tends to one. At larger scales the ellipse elongates and drifts.

For a two-phase draw the second ellipse grows at scale $h=f g+(.8-f)4s g/(4s+g)$. The large-distance fraction $f$ takes values 0, 0.3 or 0.8 with equal conditional prior probability. A secondary lobe can approach a finite reach or continue to expand. With $\rho_1,\rho_2$ the radii of the two ellipses on a bearing, the boundary of the combined deposit is

$$
\rho(t,\theta)=\sqrt{t^2-g^2+\rho_1(g,\theta)^2+\rho_2(h,\theta)^2}.
$$

Set $\rho_2=0$ for a single phase. Each angular sector receives the sum of the component areas assigned to it. All summands increase with $t$. Enclosed regions are therefore nested, and a crossing between contributions from two lobes has a continuous tangent. This is a model of the **combined thickness contours**. A phase's assigned area does not identify a separate erupted volume. A detached alternative adds a growing circular patch using the geometric union and permits the thickest bed away from the vent. The implementation counts disjoint patches and clips the union at the physical equal-area disk edge.

The prior masses are 0.20 round, 0.40 single, 0.15 turning, 0.20 two-phase and 0.05 detached. Each class divides its probability evenly across its shapes and 48 orientations. Single-phase elongations use $e=0.55,0.82,0.96$. Turning phases bend by ±70° at $e=0.91$. Two-phase bearings differ by ±75° or ±145° at $e=0.92$. The three shared migration lengths are 8, 40 and 200 km. Detached offsets are 50, 250 or 1,250 km. These choices describe a finite set of plausible deposits. Each case reports sensitivity to the class odds, migration lengths, secondary reach and outer transfer. Effective draw counts accompany each importance reweighting. [Geographic implementation](../../src/empirical_isopach/models/nested/footprints/geometry.py)

## From a contour to volume

For each point, find the first scale whose region includes it. Assign $q=\sqrt{A(t)/\pi}$ from the area of the entire clipped region. A draw has thickness $T(q)$ at that point. With distance in kilometres and thickness in centimetres, its bulk volume in km³ is $2\pi\,10^{-5}\int q\,T(q)\,dq$. Reassigning the same area to different bearings preserves this integral at fixed thinning. Fitting a different geography changes the inferred ranks of the actual measurements and can change the estimated volume substantially. The proximal decay, distal tail mixture, original robust log-thickness likelihood and transfer at the outer positive rank remain explicit. The equation also applies at geographic latitudes far from the vent through the equal-area chart. [Thinning and transfer equations](nested-isopachs.md)

The principal volume integrates the **pointwise posterior median thickness across Earth**. The conditional map-based range integrates the pointwise 25th and 75th thickness percentiles across Earth. Complete-draw integrations are retained as sampler and sensitivity checks. Each report gives the volume inside the first measured rank, between the first measured and last positive ranks, and outside that last rank as posterior fractions. Sites constrain only parts of those perimeters. The full shared support screen retains unsupported holes, discarded contours and all rejection reasons. [Support protocol](supported-isopachs.md)

## Reproduction and evidence

Use the ordinary chat intake or run the following command in a fresh destination.

~~~sh
tephra-isopachs global Mazama --geometry nested-isopachs --output results/mazama-footprints
~~~

The main fit and four buffered geographic refits use four independent populations and both distal laws. Failed partitions receive a higher-resolution attempt with up to 12,288 particles per population. The released packets identify the input and train/test rows, random seed, engine digest and all refinement attempts. Independent geographic quadrature checks coherent volumes against the analytic integral. Median-map sample size, polar-grid checks and two scrambled spatial integrals are assessed separately. Numerical convergence measures repeatability under this model. The catalogue study compares these conditional results with preserved corridors and the original whole-area model, repeats predictions after geographic deletions and evaluates known deposits. The named failure cases influenced development, so their improved figures constitute development evidence. [Reproduce the comparison](../../comparisons/nested-isopachs/geography/README.md)
