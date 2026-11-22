# Pinatubo 1991 Layer C · Bayesian nested whole-area sensitivity

[Principal corridor estimate and median isopachs](../nested-isopachs/README.md)

The median-thickness map integrates to **5.82 km³** over Earth. Its conditional 95% interval is **3.6–14.9 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 4.3–20.9 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 85.0% (conditional 95% 73.6%–93.9%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 137 accepted rows include 102 exact thicknesses, 13 lower bounds, 22 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 135 (WW-17954-1) sets the last sampled isopach rank in 100.0% of complete draws and 100.0% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent populations’ upper 95% endpoints ranged from 13.9 to 17.0 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.076. 2/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 91/102 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 5.6%. The median fraction beyond the farthest exact site is 9.9%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.21–4.15 |
| 0.01–0.1 | 1.37–6.10 |
| 0.1–1 | 1.08–5.16 |
| 1–10 | 0.47–0.76 |
| 10–100 | 0.15–0.28 |
| 100–1000 | 0.00–0.00 |
| >1000 | 0.00–0.00 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 3.6–14.9 | 24,576 |
| weak profile transfer | 3.8–15.9 | 22,664 |
| broader peak prior | 3.6–14.9 | 24,575 |
| power tail prior 80pct | 3.6–14.0 | 16,510 |
| exponential tail prior 80pct | 3.6–15.6 | 19,917 |
| detached prior 50pct | 3.6–14.9 | 24,576 |
| more distal ash prior | 3.6–15.2 | 24,150 |
| less distal ash prior | 3.6–14.9 | 24,561 |
| distal prior without Santa Maria 1902 | 3.5–13.7 | 19,867 |
| distal prior without Quizapu 1932 | 3.6–15.0 | 24,475 |
| distal prior without Novarupta 1912 | 3.6–15.1 | 24,452 |
| distal prior without Hudson 1991 | 3.6–15.1 | 24,337 |

## Compatibility with fitted bounds

Of 35 documented thickness bounds, 1 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 44 | ≤ 1 | 2.45 | 3.27% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 3.6–10.1 | 4.432 |
| 2 × sampled radius | 3.6–11.7 | 1.496 |
| 4 × sampled radius | 3.6–13.2 | 1.183 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
