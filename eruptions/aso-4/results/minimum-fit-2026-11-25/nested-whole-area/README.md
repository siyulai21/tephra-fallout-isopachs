# Aso-4 · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **2.27e+03 km³** over Earth. Its conditional 95% interval is **663.9–15,832.9 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 1,572.0–20,150.2 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 93.9% (conditional 95% 79.1%–98.4%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 87 accepted rows include 86 exact thicknesses, 0 lower bounds, 1 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 48 (37.75°, 162.72°) sets the last sampled isopach rank in 70.3% of complete draws and 52.3% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 13,478.6 to 16,715.7 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.520. 3/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 78/86 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 1.2%. The median fraction beyond the farthest exact site is 57.5%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 5.08–51.01 |
| 0.01–0.1 | 31.97–459.06 |
| 0.1–1 | 202.84–4,590.59 |
| 1–10 | 345.42–10,225.00 |
| 10–100 | 19.79–625.88 |
| 100–1000 | 0.00–12.45 |
| >1000 | 0.00–0.00 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 663.9–15,832.9 | 24,576 |
| weak profile transfer | 724.1–17,003.5 | 19,680 |
| broader peak prior | 662.9–15,833.0 | 24,564 |
| power tail prior 80pct | 661.9–15,621.4 | 16,247 |
| exponential tail prior 80pct | 665.2–16,065.6 | 20,330 |
| detached prior 50pct | 672.1–14,157.2 | 13,716 |
| more distal ash prior | 656.5–15,617.7 | 24,233 |
| less distal ash prior | 671.5–16,213.3 | 24,044 |
| distal prior without Santa Maria 1902 | 664.7–15,348.7 | 17,942 |
| distal prior without Quizapu 1932 | 679.8–16,532.2 | 20,673 |
| distal prior without Novarupta 1912 | 680.3–16,536.6 | 20,599 |
| distal prior without Hudson 1991 | 680.8–16,545.9 | 20,554 |

## Compatibility with fitted bounds

Of 1 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 635.7–15,384.0 | 1.520 |
| 2 × sampled radius | 653.7–15,729.1 | 1.520 |
| 4 × sampled radius | 660.9–15,799.3 | 1.520 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
