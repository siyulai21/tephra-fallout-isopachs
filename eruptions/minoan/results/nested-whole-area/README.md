# Minoan · Bayesian nested whole-area sensitivity

[Principal corridor estimate and median isopachs](../nested-isopachs/README.md)

**37.6 km³ of bulk fallout**, with a conditional 95% interval of **25.1–91.8 km³**. Each sample integrates one complete deposit over the Earth.

Continuing the fitted slopes freely beyond every sampled isopach rank gives 46.3 km³ [26.6–248.6]. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Reducing or increasing the transferred odds fourfold gives medians of 34.2 and 46.6 km³. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 33.4% (conditional 95% 3.4%–96.1%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 101 accepted rows include 92 exact thicknesses, 9 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 90 (42.98°, 34.01°) sets the last sampled isopach rank in 58.3% of complete draws and 70.1% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent full-data populations gave medians from 34.4 to 39.2 km³. Their upper 95% endpoints ranged from 82.6 to 97.8 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.536. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 85/92 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 6.7%. The median fraction beyond the farthest exact site is 9.5%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 1.87 [0.30–27.73] |
| 0.01–0.1 | 4.19 [1.90–25.52] |
| 0.1–1 | 9.02 [5.67–19.46] |
| 1–10 | 16.96 [11.58–30.88] |
| 10–100 | 2.94 [1.21–7.31] |
| 100–1000 | 0.00 [0.00–3.29] |
| >1000 | 0.00 [0.00–0.33] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 37.6 [25.1–91.8] | 6,144 |
| weak profile transfer | 41.1 [25.3–93.6] | 869 |
| broader peak prior | 37.8 [25.2–93.2] | 5,795 |
| power tail prior 80pct | 38.4 [24.8–88.8] | 4,504 |
| exponential tail prior 80pct | 36.9 [25.4–93.4] | 4,531 |
| detached prior 50pct | 36.5 [24.7–89.0] | 3,729 |
| more distal ash prior | 37.6 [25.5–92.2] | 4,349 |
| less distal ash prior | 37.5 [24.9–87.2] | 4,170 |
| distal prior without Santa Maria 1902 | 37.9 [25.3–95.6] | 3,397 |
| distal prior without Quizapu 1932 | 37.6 [25.1–91.1] | 6,000 |
| distal prior without Novarupta 1912 | 37.6 [25.1–91.3] | 6,005 |
| distal prior without Hudson 1991 | 37.6 [25.2–91.4] | 5,926 |

## Compatibility with fitted bounds

Of 9 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 34.5 [24.5–64.3] | 1.541 |
| 2 × sampled radius | 35.7 [24.8–66.8] | 1.537 |
| 4 × sampled radius | 36.6 [25.0–72.4] | 1.537 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
