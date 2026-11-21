# Aso-4 · Bayesian nested isopachs

**2,513.9 km³ of bulk fallout**, with a conditional 95% interval of **648.8–15,969.9 km³**. Each sample integrates one complete deposit over the Earth.

Continuing the fitted slopes freely beyond every sampled isopach rank gives 6,493.5 km³ [1,354.2–20,663.7]. The principal fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Reducing or increasing the transferred odds fourfold gives medians of 1,516.1 and 4,765.3 km³. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 93.9% (conditional 95% 80.4%–98.5%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 87 accepted rows include 83 exact thicknesses, 3 lower bounds, 1 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 48 (37.75°, 162.72°) sets the last sampled isopach rank in 70.1% of complete draws and 59.5% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent full-data populations gave medians from 2,294.5 to 2,749.9 km³. Their upper 95% endpoints ranged from 13,732.1 to 16,457.3 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.540. 3/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 75/83 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 1.2%. The median fraction beyond the farthest exact site is 55.1%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 51.01 [4.93–51.01] |
| 0.01–0.1 | 281.58 [31.45–459.06] |
| 0.1–1 | 993.56 [198.98–4,590.59] |
| 1–10 | 1,033.24 [332.88–10,446.39] |
| 10–100 | 53.62 [18.75–525.11] |
| 100–1000 | 0.00 [0.00–11.65] |
| >1000 | 0.00 [0.00–0.00] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 2,513.9 [648.8–15,969.9] | 24,576 |
| weak profile transfer | 2,922.4 [719.9–16,916.2] | 19,409 |
| broader peak prior | 2,510.1 [647.8–15,973.7] | 24,561 |
| power tail prior 80pct | 2,480.2 [648.3–15,300.9] | 16,281 |
| exponential tail prior 80pct | 2,536.0 [648.9–16,308.0] | 20,273 |
| detached prior 50pct | 3,391.8 [651.9–14,471.9] | 13,723 |
| more distal ash prior | 2,466.5 [646.1–15,849.6] | 24,254 |
| less distal ash prior | 2,564.4 [658.5–16,184.0] | 24,024 |
| distal prior without Santa Maria 1902 | 2,564.4 [633.7–15,717.3] | 20,736 |
| distal prior without Quizapu 1932 | 2,629.1 [669.4–16,380.0] | 21,442 |
| distal prior without Novarupta 1912 | 2,630.3 [669.8–16,384.4] | 21,365 |
| distal prior without Hudson 1991 | 2,631.5 [671.0–16,410.3] | 21,261 |

## Compatibility with fitted bounds

Of 4 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 1,845.0 [623.2–15,540.0] | 1.539 |
| 2 × sampled radius | 2,172.9 [641.3–15,853.9] | 1.540 |
| 4 × sampled radius | 2,392.6 [646.2–15,932.7] | 1.540 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
