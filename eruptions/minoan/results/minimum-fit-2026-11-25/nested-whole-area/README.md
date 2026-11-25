# Minoan · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **33 km³** over Earth. Its conditional 95% interval is **24.2–112.0 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 25.3–212.9 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 32.5% (conditional 95% 2.8%–88.0%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 101 accepted rows include 101 exact thicknesses, 0 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 90 (42.98°, 34.01°) sets the last sampled isopach rank in 69.8% of complete draws and 83.1% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 93.5 to 123.9 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.657. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 94/101 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 7.1%. The median fraction beyond the farthest exact site is 13.4%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.61–33.91 |
| 0.01–0.1 | 2.06–32.19 |
| 0.1–1 | 5.96–21.50 |
| 1–10 | 11.63–36.07 |
| 10–100 | 0.39–5.07 |
| 100–1000 | 0.00–0.00 |
| >1000 | 0.00–0.00 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 24.2–112.0 | 3,072 |
| weak profile transfer | 22.0–114.2 | 496 |
| broader peak prior | 24.2–112.7 | 3,070 |
| power tail prior 80pct | 24.2–99.1 | 2,565 |
| exponential tail prior 80pct | 24.4–124.2 | 2,017 |
| detached prior 50pct | 24.1–100.4 | 2,452 |
| more distal ash prior | 24.4–125.9 | 2,057 |
| less distal ash prior | 24.3–105.6 | 2,270 |
| distal prior without Santa Maria 1902 | 24.0–108.1 | 1,618 |
| distal prior without Quizapu 1932 | 24.3–110.7 | 2,929 |
| distal prior without Novarupta 1912 | 24.3–111.8 | 2,916 |
| distal prior without Hudson 1991 | 24.4–112.7 | 2,837 |

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 23.3–69.4 | 1.658 |
| 2 × sampled radius | 23.8–75.8 | 1.656 |
| 4 × sampled radius | 24.0–85.8 | 1.657 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
