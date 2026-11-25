# Sakurajima Satsuma (P14) · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **10.3 km³** over Earth. Its conditional 95% interval is **6.1–56.3 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 6.2–115.2 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 81.1% (conditional 95% 2.5%–97.9%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 57 accepted rows include 55 exact thicknesses, 0 lower bounds, 2 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 57 (MD98-2195-A) sets the last sampled isopach rank in 43.7% of complete draws and 32.7% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 54.2 to 60.7 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 0.887. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 55/55 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 1.6%. The median fraction beyond the farthest exact site is 43.7%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.03–8.99 |
| 0.01–0.1 | 0.16–14.63 |
| 0.1–1 | 0.98–24.39 |
| 1–10 | 3.00–10.97 |
| 10–100 | 1.13–2.69 |
| 100–1000 | 0.00–0.37 |
| >1000 | 0.00–0.00 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 6.1–56.3 | 12,288 |
| weak profile transfer | 6.1–56.5 | 4,838 |
| broader peak prior | 6.1–56.3 | 12,261 |
| power tail prior 80pct | 6.2–55.3 | 8,698 |
| exponential tail prior 80pct | 6.1–57.5 | 9,394 |
| detached prior 50pct | 6.2–61.0 | 8,075 |
| more distal ash prior | 6.1–55.6 | 9,904 |
| less distal ash prior | 6.1–58.1 | 7,406 |
| distal prior without Santa Maria 1902 | 6.2–57.0 | 7,061 |
| distal prior without Quizapu 1932 | 6.1–57.4 | 11,558 |
| distal prior without Novarupta 1912 | 6.1–57.4 | 11,527 |
| distal prior without Hudson 1991 | 6.1–57.0 | 11,346 |

## Compatibility with fitted bounds

Of 2 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 5.9–14.8 | 0.887 |
| 2 × sampled radius | 6.0–19.4 | 0.887 |
| 4 × sampled radius | 6.1–26.9 | 0.887 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
