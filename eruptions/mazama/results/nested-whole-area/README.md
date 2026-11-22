# Mazama · Bayesian nested whole-area sensitivity

[Principal corridor estimate and median isopachs](../nested-isopachs/README.md)

The median-thickness map integrates to **355 km³** over Earth. Its conditional 95% interval is **162.2–1,663.6 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 155.1–1,731.1 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 73.3% (conditional 95% 2.4%–98.2%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 45 accepted rows include 45 exact thicknesses, 0 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 5 (54.69°, -112.51°) sets the last sampled isopach rank in 59.9% of complete draws and 17.6% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent populations’ upper 95% endpoints ranged from 1,575.4 to 1,731.9 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.503. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 41/45 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 5.4%. The median fraction beyond the farthest exact site is 15.3%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.58–51.01 |
| 0.01–0.1 | 3.54–389.86 |
| 0.1–1 | 19.23–644.82 |
| 1–10 | 62.06–465.89 |
| 10–100 | 48.97–358.20 |
| 100–1000 | 0.00–89.14 |
| >1000 | 0.00–2.03 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 162.2–1,663.6 | 24,576 |
| weak profile transfer | 161.9–1,857.0 | 3,658 |
| broader peak prior | 162.1–1,647.5 | 20,500 |
| power tail prior 80pct | 164.9–1,632.4 | 16,424 |
| exponential tail prior 80pct | 161.3–1,687.7 | 20,046 |
| detached prior 50pct | 162.2–1,662.4 | 23,072 |
| more distal ash prior | 162.3–1,680.3 | 18,900 |
| less distal ash prior | 158.1–1,558.1 | 14,938 |
| distal prior without Santa Maria 1902 | 165.3–1,715.4 | 14,276 |
| distal prior without Quizapu 1932 | 161.8–1,649.3 | 23,358 |
| distal prior without Novarupta 1912 | 161.8–1,650.0 | 23,270 |
| distal prior without Hudson 1991 | 161.9–1,652.6 | 22,597 |

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 154.3–1,063.5 | 1.501 |
| 2 × sampled radius | 158.5–1,142.1 | 1.502 |
| 4 × sampled radius | 160.9–1,288.1 | 1.503 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
