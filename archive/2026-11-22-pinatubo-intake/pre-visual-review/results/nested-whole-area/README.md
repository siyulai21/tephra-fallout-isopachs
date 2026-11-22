# Pinatubo 1991 (Layer C) · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **10.9 km³** over Earth. Its conditional 95% interval is **9.4–27.8 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 9.2–51.9 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 22.7% (conditional 95% 0.7%–82.8%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 194 accepted rows include 140 exact thicknesses, 26 lower bounds, 28 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 166 (WW-17954-1) sets the last sampled isopach rank in 96.1% of complete draws and 97.4% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 23.4 to 33.8 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 2.057. 3/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 101/140 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 6.7%. The median fraction beyond the farthest exact site is 9.8%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.21–12.01 |
| 0.01–0.1 | 0.66–7.04 |
| 0.1–1 | 2.70–3.58 |
| 1–10 | 4.82–6.35 |
| 10–100 | 0.20–0.55 |
| 100–1000 | 0.00–0.00 |
| >1000 | 0.00–0.00 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 9.4–27.8 | 3,072 |
| weak profile transfer | 9.6–25.5 | 332 |
| broader peak prior | 9.4–27.8 | 3,072 |
| power tail prior 80pct | 9.4–25.3 | 2,300 |
| exponential tail prior 80pct | 9.5–29.6 | 2,219 |
| detached prior 50pct | 9.4–27.8 | 3,072 |
| more distal ash prior | 9.4–28.3 | 1,958 |
| less distal ash prior | 9.4–26.0 | 2,234 |
| distal prior without Santa Maria 1902 | 9.5–28.8 | 1,288 |
| distal prior without Quizapu 1932 | 9.4–27.4 | 2,950 |
| distal prior without Novarupta 1912 | 9.4–27.4 | 2,945 |
| distal prior without Hudson 1991 | 9.4–27.7 | 2,912 |

## Compatibility with fitted bounds

Of 54 documented thickness bounds, 1 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 35 | ≥ 10.8 | 3.94 | 3.76% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 9.2–11.8 | 4.814 |
| 2 × sampled radius | 9.3–13.2 | 2.562 |
| 4 × sampled radius | 9.4–15.9 | 2.188 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
