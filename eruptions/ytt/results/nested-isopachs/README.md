# YTT · Bayesian nested isopachs

**1,543.0 km³ of bulk fallout**, with a conditional 95% interval of **1,048.5–3,339.5 km³**. Each sample integrates one complete deposit over the Earth.

Continuing the fitted slopes freely beyond every sampled isopach rank gives 1,538.2 km³ [1,046.0–3,202.7]. The principal fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Reducing or increasing the transferred odds fourfold gives medians of 1,535.5 and 1,552.8 km³. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 10.4% (conditional 95% 1.6%–52.2%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 46 accepted rows include 46 exact thicknesses, 0 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 38 (-3.32°, 37.70°) sets the last sampled isopach rank in 99.7% of complete draws and 95.6% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent full-data populations gave medians from 1,411.3 to 2,977.9 km³. Their upper 95% endpoints ranged from 2,667.3 to 3,429.4 km³. The unmodified thinning profile retained a warning and the continued global volume retained a warning. [Population estimates and refinement history](full.json)

The matched geographic score is 1.507. 3/4 training-fold fits passed the population check. The full-fit population check retained a warning. Direct spatial integration passed.

At withheld exact sites, 43/46 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 13.2%. The median fraction beyond the farthest exact site is 0.9%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 22.47 [5.44–51.01] |
| 0.01–0.1 | 48.47 [26.11–193.10] |
| 0.1–1 | 208.13 [168.33–537.22] |
| 1–10 | 949.57 [692.24–2,065.05] |
| 10–100 | 309.55 [65.42–706.84] |
| 100–1000 | 0.00 [0.00–0.00] |
| >1000 | 0.00 [0.00–0.00] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 1,543.0 [1,048.5–3,339.5] | 49,152 |
| weak profile transfer | 1,534.9 [1,048.5–3,314.8] | 7,723 |
| broader peak prior | 1,544.1 [1,045.9–3,343.3] | 48,780 |
| power tail prior 80pct | 1,462.5 [1,017.7–3,228.5] | 41,449 |
| exponential tail prior 80pct | 2,816.2 [1,116.9–3,413.1] | 32,071 |
| detached prior 50pct | 2,916.3 [1,046.2–3,438.1] | 29,377 |
| more distal ash prior | 1,594.7 [1,059.9–3,330.6] | 35,872 |
| less distal ash prior | 1,517.5 [1,029.4–3,328.3] | 46,091 |
| distal prior without Santa Maria 1902 | 1,694.4 [1,077.4–3,332.3] | 14,966 |
| distal prior without Quizapu 1932 | 1,531.3 [1,044.8–3,332.8] | 48,466 |
| distal prior without Novarupta 1912 | 1,530.9 [1,044.5–3,331.9] | 48,348 |
| distal prior without Hudson 1991 | 1,531.2 [1,045.3–3,329.2] | 47,924 |

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 1,538.2 [1,044.3–3,293.8] | 1.501 |
| 2 × sampled radius | 1,541.1 [1,047.2–3,324.7] | 1.505 |
| 4 × sampled radius | 1,542.5 [1,048.2–3,336.1] | 1.506 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
