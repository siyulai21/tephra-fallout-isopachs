# Tambora · Bayesian nested whole-area sensitivity

[Principal corridor estimate and median isopachs](../nested-isopachs/README.md)

The median-thickness map integrates to **123 km³** over Earth. Its conditional 95% interval is **96.1–290.7 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 101.9–460.2 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 26.8% (conditional 95% 2.3%–96.5%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 26 accepted rows include 26 exact thicknesses, 0 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 27 (0.78°, 127.83°) sets the last sampled isopach rank in 30.6% of complete draws and 31.2% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv) · [Refit with the controlling site and its local cluster withheld](edge-deletion/README.md)

The independent populations’ upper 95% endpoints ranged from 268.8 to 304.5 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.368. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 26/26 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 8.0%. The median fraction beyond the farthest exact site is 5.1%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 1.25–33.32 |
| 0.01–0.1 | 4.93–43.62 |
| 0.1–1 | 12.75–61.86 |
| 1–10 | 35.19–133.90 |
| 10–100 | 18.77–82.47 |
| 100–1000 | 0.00–16.62 |
| >1000 | 0.00–0.24 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 96.1–290.7 | 6,144 |
| weak profile transfer | 97.7–294.8 | 722 |
| broader peak prior | 96.1–300.2 | 5,680 |
| power tail prior 80pct | 96.1–276.8 | 4,999 |
| exponential tail prior 80pct | 96.5–311.6 | 4,114 |
| detached prior 50pct | 99.0–318.4 | 3,462 |
| more distal ash prior | 96.2–306.1 | 4,158 |
| less distal ash prior | 96.3–282.5 | 4,438 |
| distal prior without Santa Maria 1902 | 95.9–282.9 | 3,325 |
| distal prior without Quizapu 1932 | 96.2–293.6 | 5,969 |
| distal prior without Novarupta 1912 | 96.2–294.2 | 5,961 |
| distal prior without Hudson 1991 | 96.2–298.8 | 5,820 |

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 95.5–279.4 | 1.372 |
| 2 × sampled radius | 96.0–282.8 | 1.370 |
| 4 × sampled radius | 96.0–285.8 | 1.369 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
