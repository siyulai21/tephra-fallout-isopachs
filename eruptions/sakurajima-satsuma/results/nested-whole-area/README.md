# Sakurajima Satsuma (P14) · Bayesian nested whole-area sensitivity

[Principal corridor estimate and median isopachs](../nested-isopachs/README.md)

**12.5 km³ of bulk fallout**, with a conditional 95% interval of **7.0–60.2 km³**. Each sample integrates one complete deposit over the Earth.

Continuing the fitted slopes freely beyond every sampled isopach rank gives 12.8 km³ [6.6–402.0]. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Reducing or increasing the transferred odds fourfold gives medians of 9.0 and 26.3 km³. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 59.5% (conditional 95% 1.3%–97.7%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 57 accepted rows include 50 exact thicknesses, 5 lower bounds, 2 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 49 (白ヶ野第1・第4遺跡) sets the last sampled isopach rank in 82.4% of complete draws and 88.0% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent full-data populations gave medians from 12.1 to 13.2 km³. Their upper 95% endpoints ranged from 56.7 to 63.7 km³. The unmodified thinning profile retained a warning and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 0.742. 1/4 training-fold fits passed the population check. The full-fit population check retained a warning. Direct spatial integration passed.

At withheld exact sites, 49/50 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 1.4%. The median fraction beyond the farthest exact site is 45.9%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 0.67 [0.03–10.92] |
| 0.01–0.1 | 1.43 [0.19–16.90] |
| 0.1–1 | 3.18 [1.09–25.54] |
| 1–10 | 4.54 [3.17–10.38] |
| 10–100 | 2.00 [1.44–2.87] |
| 100–1000 | 0.00 [0.00–0.25] |
| >1000 | 0.00 [0.00–0.00] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 12.5 [7.0–60.2] | 49,152 |
| weak profile transfer | 12.6 [7.1–60.7] | 8,981 |
| broader peak prior | 12.4 [7.0–60.1] | 48,924 |
| power tail prior 80pct | 12.7 [7.0–58.9] | 35,441 |
| exponential tail prior 80pct | 12.3 [6.9–61.6] | 36,864 |
| detached prior 50pct | 12.5 [7.0–60.5] | 44,717 |
| more distal ash prior | 12.5 [7.0–60.8] | 36,772 |
| less distal ash prior | 12.2 [6.9–59.6] | 30,321 |
| distal prior without Santa Maria 1902 | 13.0 [7.0–61.1] | 25,786 |
| distal prior without Quizapu 1932 | 12.5 [7.0–60.2] | 45,315 |
| distal prior without Novarupta 1912 | 12.5 [7.0–60.2] | 45,017 |
| distal prior without Hudson 1991 | 12.5 [7.0–60.3] | 43,439 |

## Compatibility with fitted bounds

Of 7 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 9.2 [6.6–15.8] | 0.741 |
| 2 × sampled radius | 10.3 [6.8–20.5] | 0.741 |
| 4 × sampled radius | 11.2 [6.9–28.5] | 0.742 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
