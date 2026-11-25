# Ilopango TBJ (total primary airfall) · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **42.3 km³** over Earth. Its conditional 95% interval is **30.5–92.5 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 33.9–126.9 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 93.5% (conditional 95% 88.3%–97.2%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 168 accepted rows include 149 exact thicknesses, 0 lower bounds, 5 upper bounds and 14 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 91 (Osa margin, Dull M12) sets the last sampled isopach rank in 100.0% of complete draws and 99.4% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 86.2 to 97.7 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.568. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 137/149 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 5.5%. The median fraction beyond the farthest exact site is 5.8%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.20–15.26 |
| 0.01–0.1 | 1.32–20.25 |
| 0.1–1 | 7.47–26.13 |
| 1–10 | 13.18–34.52 |
| 10–100 | 4.51–9.77 |
| 100–1000 | 0.10–1.70 |
| >1000 | 0.00–0.01 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 30.5–92.5 | 3,072 |
| weak profile transfer | 31.0–96.0 | 2,431 |
| broader peak prior | 30.4–91.9 | 3,061 |
| power tail prior 80pct | 30.3–93.4 | 2,802 |
| exponential tail prior 80pct | 30.6–91.4 | 1,980 |
| detached prior 50pct | 30.4–92.9 | 3,072 |
| more distal ash prior | 30.6–93.3 | 3,063 |
| less distal ash prior | 30.5–92.9 | 3,066 |
| distal prior without Santa Maria 1902 | 29.7–90.2 | 2,783 |
| distal prior without Quizapu 1932 | 30.6–93.3 | 3,046 |
| distal prior without Novarupta 1912 | 30.6–93.4 | 3,042 |
| distal prior without Hudson 1991 | 30.7–93.4 | 3,030 |

## Compatibility with fitted bounds

Of 19 documented thickness bounds, 3 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 81 | 16–18 | 22.9 | 4.86% |
| Row 165 | 45–50 | 23 | 3.27% |
| Row 70 | 0.2–0.3 | 2.12 | 1.93% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 30.1–67.4 | 1.571 |
| 2 × sampled radius | 30.3–72.2 | 1.569 |
| 4 × sampled radius | 30.4–79.7 | 1.568 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
