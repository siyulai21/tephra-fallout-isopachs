# Pinatubo 1991 (Layer C) · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **10.9 km³** over Earth. Its conditional 95% interval is **9.5–28.6 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 9.3–60.3 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 30.0% (conditional 95% 0.3%–85.1%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 193 accepted rows include 139 exact thicknesses, 26 lower bounds, 28 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 166 (WW-17954-1) sets the last sampled isopach rank in 89.5% of complete draws and 97.1% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 27.1 to 29.9 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 2.020. 2/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 98/139 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 5.8%. The median fraction beyond the farthest exact site is 9.6%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.21–12.98 |
| 0.01–0.1 | 0.61–7.22 |
| 0.1–1 | 2.54–3.62 |
| 1–10 | 4.91–6.98 |
| 10–100 | 0.20–0.77 |
| 100–1000 | 0.00–0.00 |
| >1000 | 0.00–0.00 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 9.5–28.6 | 6,144 |
| weak profile transfer | 9.8–29.0 | 970 |
| broader peak prior | 9.5–28.7 | 6,143 |
| power tail prior 80pct | 9.5–28.2 | 4,289 |
| exponential tail prior 80pct | 9.5–29.2 | 4,767 |
| detached prior 50pct | 9.5–28.7 | 6,144 |
| more distal ash prior | 9.5–29.6 | 4,012 |
| less distal ash prior | 9.5–28.3 | 4,286 |
| distal prior without Santa Maria 1902 | 9.5–29.0 | 3,092 |
| distal prior without Quizapu 1932 | 9.5–29.0 | 5,792 |
| distal prior without Novarupta 1912 | 9.5–29.0 | 5,755 |
| distal prior without Hudson 1991 | 9.5–29.2 | 5,586 |

## Compatibility with fitted bounds

Of 54 documented thickness bounds, 2 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 184 | ≤ 2 | 5.61 | 4.55% |
| Row 35 | ≥ 10.8 | 4.01 | 3.87% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 9.3–12.1 | 4.734 |
| 2 × sampled radius | 9.4–13.4 | 2.536 |
| 4 × sampled radius | 9.5–16.2 | 2.155 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
