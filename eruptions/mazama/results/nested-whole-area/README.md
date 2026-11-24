# Mazama · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **325 km³** over Earth. Its conditional 95% interval is **251.3–876.9 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 267.8–987.1 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 36.4% (conditional 95% 15.5%–59.5%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 244 accepted rows include 58 exact thicknesses, 153 lower bounds, 33 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 241 (Fish Lake Valley, Furnace Creek hand auger) sets the last sampled isopach rank in 81.7% of complete draws and 68.2% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 756.4 to 954.5 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 0.858. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 42/58 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 8.0%. The median fraction beyond the farthest exact site is 14.5%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.35–4.22 |
| 0.01–0.1 | 2.88–23.48 |
| 0.1–1 | 24.01–124.23 |
| 1–10 | 49.03–131.65 |
| 10–100 | 19.36–69.16 |
| 100–1000 | 62.74–171.94 |
| >1000 | 22.11–513.31 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 251.3–876.9 | 3,072 |
| weak profile transfer | 246.1–1,080.7 | 1,435 |
| broader peak prior | 259.4–973.3 | 2,882 |
| power tail prior 80pct | 253.6–839.4 | 2,416 |
| exponential tail prior 80pct | 248.6–897.0 | 2,118 |
| detached prior 50pct | 246.1–876.7 | 1,756 |
| more distal ash prior | 251.0–786.8 | 2,609 |
| less distal ash prior | 256.1–1,080.8 | 2,617 |
| distal prior without Santa Maria 1902 | 248.7–706.2 | 1,733 |
| distal prior without Quizapu 1932 | 251.9–907.0 | 3,041 |
| distal prior without Novarupta 1912 | 251.9–907.0 | 3,046 |
| distal prior without Hudson 1991 | 251.9–896.1 | 3,062 |

## Compatibility with fitted bounds

Of 186 documented thickness bounds, 1 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 58 | ≥ 60 | 5.26 | 3.03% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 251.3–876.9 | 0.859 |
| 2 × sampled radius | 251.3–876.9 | 0.858 |
| 4 × sampled radius | 251.3–876.9 | 0.858 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
