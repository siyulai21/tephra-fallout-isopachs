# Directional isopachs in unsampled sectors

A short ray with a measured ash bed establishes fallout along that bearing at a particular distance. It provides limited evidence about fallout hundreds of kilometres farther away. An empty sector can contain either little ash or an undetected fan. These possibilities lead to different contours and global volumes.

## What the maps show

**The principal directional map traces pointwise median thickness with uncertain fan direction.** Integrating that quantitative field over the Earth gives the headline bulk volume. The six-structure fixed-direction reconstruction and visibility-conditioned archive scenarios are reported as sensitivities. The case-home map draws a segment solid only where exact observations on a retained triangle bracket its level and at least 90% of the posterior crosses that level on the triangle. Each segment is clipped to that triangle. Other sections of the very same contour appear dashed, including gaps inside the convex hull. This local check permits a measured fragment even when a complete closed contour fails the shared support screen. The source-centred detail pages apply the same display rule to both directional assumptions.

The exceedance map provides a second reading. A point on the 90% line for 10 cm has a model-conditional 90% probability of reaching 10 cm. That line can miss a real deposit in an unsampled sector. Source-connected reach is the first crossing along each of eight compass rays. Sector area also counts detached patches. The plotting window never enters the volume or area integral.

## One uncertain fan direction per deposit

Start with a posterior draw of the positive thickness field \(T_0\) from the six existing fan structures. Work in the vent-centred, equal-area authalic chart, with chord radius \(r\) and clockwise bearing \(\theta\). Build a finite segment from the vent to each measured positive or lower-bound location. Repeated coordinates supply one segment. A quantitative upper limit, including a defensible interpreted limit, anchors its own location. A site where review establishes only arrival remains an occurrence check. [Source review and principal bounds](sites-without-measured-thickness.md)

Let \(d\) be the distance from the location to the nearest segment. For this computation, cap the radial coordinate at the longest observed positive-site radius \(R\). This preserves a distinction between a long, sampled bearing and an empty bearing even beyond the last site. For an upper limit, also take the minimum with the actual distance to its location. Let \(s\) be the median nearest-neighbour distance among distinct exact sites and \(b\) the sum of the fitted local interpolation weights. The exposure to unsampled directions is

\[
h=(1-b)\frac{d^2}{d^2+s^2+\bigl(\min(r,R)\sin(22.5^\circ)\bigr)^2}.
\]

The half-sector angle sets a common geometric scale. Every accepted quantitative observation, the vent and the interiors of measured triangles have zero exposure. A ray can protect a well-sampled bearing without filling its unsampled neighbours. The transition collar interpolates continuously across the edge of a retained triangle.

For each complete deposit, choose one orientation \(\Phi\) uniformly on the circle and one concentration \(\kappa\) from 0, 2, 4 and 8. Use them at every location in that deposit:

\[
T=T_0\,\frac{\exp\{\kappa h\cos(\theta-\Phi)\}}{I_0(\kappa h)}.
\]

Here \(I_0\) is the modified Bessel function. At fixed \(T_0,\kappa,h,\theta\), integrating over \(\Phi\) gives exactly \(E[T]=T_0\). The median often retreats where \(h\) is large. Individual deposits can still send substantial ash into the gap. The normalization preserves the conditional mean at fixed structure weights. Individual sampled masses and pointwise medians can change. The quantitative observations leave the new orientation unidentified within an eruption. The quantitative likelihood at the training sites is identical to the base likelihood, allowing exact reuse of the audited base posterior with independent draws of orientation and concentration. This term represents uncertainty about fan direction. A real eruption can have more than one unmapped fan, a different angular shape, or an unobserved proximal maximum.

## Calibration and validation

Every target excludes its own eruption, any identical input digest and the same vent from calibration. Training geometry is rebuilt for each buffered geographic holdout and withheld source quadrant. The structure weights are learned first at zero concentration. The concentration weights are then learned on the other eruptions with those structure weights. Donor eruptions, available deletion designs and passing folds within each design each have equal influence at their respective levels. One eruption-equivalent of regularization keeps rare structures in the mixture. The fixed-direction control uses the identical structure weights. The historical headline uses the earlier geographic-only calibration.

An exact observation contributes its predictive log density. A bound contributes its one-sided probability. Single-site held-out likelihoods integrate over orientation by deterministic cosine quadrature with 16 through 128 nodes. Successive per-row log scores must differ by less than 0.001 for a fold to enter calibration. Poorly mixed or inestimable folds remain visible in the validation record. Passing-only and all-estimable summaries answer different questions.

The project uses uncertain fan direction as its primary assumption in unsampled sectors. The matched control remains important for interpreting that choice. With the current B-Tm bounds, the published fits give lower pooled negative log scores for uncertain direction on both geographic and whole-quadrant holdouts. The original comparison on 68 B-Tm sites gave a higher uncertain-direction score for quadrants. The change illustrates sensitivity to which sites can test an unsampled sector. The real cases have preferential sampling at known ash sites. Scores at those sites cannot determine the probability of a macroscopic bed at an unvisited location. A pair of synthetic deposits with identical observations and different hidden lobes demonstrates the limitation directly. [Cross-eruption findings](../../comparisons/directional/README.md)

## Volume, site evidence and uncertainty

Every member covers the Earth. Two independently scrambled, nested, equal-area Sobol rules integrate thickness to the antipode, including proximal and distal tails. The central bulk figure integrates the pointwise median. The conditional interval integrates each coherent deposit first, then takes its weighted 2.5th and 97.5th percentiles.  Absolute differences across 40 support-class and sector cells test spatial convergence. The nested limit is 1.5% and the between-scramble limit is 2.5%. Joint volume quantiles must agree between scrambles to 2.5%. Separate checks test posterior-split median volume (2.5%), level-area disagreement (10% at ≥64 effective union nodes) and joint percentiles (10%). The packet can grow from 64 to 512 draws per chain. Failures at the ceiling are reported.

A core with a source-supported bound on local fallout enters the principal quantitative likelihood once. A separate visibility scenario can test an additional core whose recognition scale requires a hypothetical transfer. For an assumed recognizable local thickness \(u_i\), the factor for a reading with no visible bed is \(P(Y_i<u_i\mid T)\) under the fitted local scatter. Apply the product to each whole deposit and normalize its weight. The archive inventory records inspection, dependence and provenance. The effects of conditioning on one archive can be checked at another that was held out. The numerical record gives effective sample size, maximum weight, sensitivity to an independently randomized packet, and an analytic integration over orientation for each unconditional site median. Archive-conditioned global volumes are scenario estimates. Visibility, post-depositional transport, structural assumptions and the priors limit what the data identify.

## Reproduce the comparison

A normal eruption run produces the estimates and maps in the directional results folder. To reproduce the study from a fresh ignored destination:

~~~bash
python -m benchmarks.azimuth_continuation fit --work results/extent/fits --output results/extent/reports
python -m benchmarks.azimuth_continuation calibrate --work results/extent/fits --output results/extent/calibration.json
python -m benchmarks.azimuth_continuation report --work results/extent/fits --output results/extent/reports
python -m benchmarks.azimuth_continuation fit --scenarios smooth multi_lobe missing_sector missing_core matched_two_fans diffuse_global_tail invisible_sector --work results/extent/fits --output results/extent/synthetic
python -m benchmarks.azimuth_continuation synthetic --work results/extent/fits --output results/extent/synthetic
~~~

The report command uses the packaged, versioned calibration file. Compare the regenerated calibration to that file before interpreting differences. Each field packet stores structure parameters, original draw indices, concentration indices, orientation, weights and complete-deposit volumes. [Calibration file](../../src/empirical_isopach/assets/azimuth-calibration.json)

To compare the current case reports and the retained fourteen synthetic trials without refitting, run `python -m benchmarks.azimuth_summary --output results/directional-summary`.
