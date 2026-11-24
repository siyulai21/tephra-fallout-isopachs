# Mazama · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **210 km³** over Earth. Its conditional 95% interval is **136.0–750.3 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 147.4–837.7 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 17.8% (conditional 95% 6.8%–31.8%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 271 accepted rows include 31 exact thicknesses, 137 lower bounds, 103 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 26 (Lofty Lake) sets the last sampled isopach rank in 78.8% of complete draws and 74.8% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 622.2 to 984.5 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 0.334. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 29/31 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 19.5%. The median fraction beyond the farthest exact site is 3.5%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.32–2.11 |
| 0.01–0.1 | 2.79–8.68 |
| 0.1–1 | 16.94–48.23 |
| 1–10 | 10.17–32.03 |
| 10–100 | 15.79–38.73 |
| 100–1000 | 50.21–152.94 |
| >1000 | 19.07–520.77 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 136.0–750.3 | 12,288 |
| weak profile transfer | 134.5–798.4 | 11,932 |
| broader peak prior | 142.1–984.0 | 11,515 |
| power tail prior 80pct | 139.3–732.3 | 8,354 |
| exponential tail prior 80pct | 134.3–761.3 | 9,819 |
| detached prior 50pct | 138.7–774.3 | 7,271 |
| more distal ash prior | 134.2–711.4 | 11,075 |
| less distal ash prior | 142.3–912.3 | 11,021 |
| distal prior without Santa Maria 1902 | 130.1–599.8 | 4,294 |
| distal prior without Quizapu 1932 | 137.6–812.1 | 12,166 |
| distal prior without Novarupta 1912 | 137.7–812.2 | 12,165 |
| distal prior without Hudson 1991 | 137.4–819.3 | 12,187 |

## Compatibility with fitted bounds

Of 240 documented thickness bounds, 1 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 9 | ≥ 30 | 2.79 | 1.20% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 135.8–750.2 | 0.334 |
| 2 × sampled radius | 135.9–750.3 | 0.334 |
| 4 × sampled radius | 136.0–750.3 | 0.334 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
