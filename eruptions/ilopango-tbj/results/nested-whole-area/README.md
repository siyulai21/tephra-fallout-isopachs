# Ilopango TBJ (Unit G) · Bayesian nested whole-area sensitivity

[Principal corridor estimate and median isopachs](../nested-isopachs/README.md)

**50.5 km³ of bulk fallout**, with a conditional 95% interval of **37.5–181.8 km³**. Each sample integrates one complete deposit over the Earth.

Continuing the fitted slopes freely beyond every sampled isopach rank gives 87.2 km³ [47.9–208.9]. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Reducing or increasing the transferred odds fourfold gives medians of 46.1 and 64.5 km³. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 91.5% (conditional 95% 78.7%–97.5%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 56 accepted rows include 52 exact thicknesses, 4 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 56 (OC001) sets the last sampled isopach rank in 94.9% of complete draws and 61.3% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent full-data populations gave medians from 49.3 to 51.8 km³. Their upper 95% endpoints ranged from 133.4 to 230.4 km³. The unmodified thinning profile retained a warning and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.159. 1/4 training-fold fits passed the population check. The full-fit population check retained a warning. Direct spatial integration passed.

At withheld exact sites, 49/52 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 3.4%. The median fraction beyond the farthest exact site is 11.3%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 0.55 [0.07–19.64] |
| 0.01–0.1 | 1.34 [0.44–32.47] |
| 0.1–1 | 5.94 [3.72–63.20] |
| 1–10 | 24.34 [18.60–64.25] |
| 10–100 | 13.84 [9.29–20.83] |
| 100–1000 | 2.24 [0.44–4.47] |
| >1000 | 0.01 [0.00–3.47] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 50.5 [37.5–181.8] | 49,152 |
| weak profile transfer | 51.2 [37.7–205.2] | 32,465 |
| broader peak prior | 50.3 [37.5–165.6] | 47,512 |
| power tail prior 80pct | 49.6 [37.2–165.9] | 42,537 |
| exponential tail prior 80pct | 52.3 [38.3–210.7] | 31,642 |
| detached prior 50pct | 50.5 [37.5–182.8] | 49,056 |
| more distal ash prior | 50.4 [37.5–178.7] | 48,796 |
| less distal ash prior | 50.5 [37.5–182.0] | 49,047 |
| distal prior without Santa Maria 1902 | 51.0 [37.4–200.4] | 36,758 |
| distal prior without Quizapu 1932 | 50.4 [37.5–181.2] | 48,710 |
| distal prior without Novarupta 1912 | 50.4 [37.5–181.1] | 48,650 |
| distal prior without Hudson 1991 | 50.4 [37.5–180.4] | 48,409 |

## Compatibility with fitted bounds

Of 4 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 48.8 [37.2–111.3] | 1.159 |
| 2 × sampled radius | 49.6 [37.3–117.3] | 1.159 |
| 4 × sampled radius | 50.0 [37.4–133.8] | 1.159 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
