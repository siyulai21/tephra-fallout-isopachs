# Tambora · Bayesian nested isopachs

**149.1 km³ of bulk fallout**, with a conditional 95% interval of **101.4–315.3 km³**. Each sample integrates one complete deposit over the Earth.

The 26 accepted rows include 26 exact thicknesses, 0 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 1.348. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 26/26 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 7.9%. The median fraction beyond the farthest exact site is 4.7%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 4.02 [0.84–51.01] |
| 0.01–0.1 | 9.73 [4.83–103.23] |
| 0.1–1 | 24.92 [14.11–64.52] |
| 1–10 | 63.40 [36.77–143.90] |
| 10–100 | 40.05 [18.10–81.00] |
| 100–1000 | 0.00 [0.00–4.62] |
| >1000 | 0.00 [0.00–0.00] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 149.1 [101.4–315.3] | 3,072 |
| weak profile transfer | 151.3 [102.5–335.2] | 479 |
| broader peak prior | 149.2 [101.4–316.6] | 3,068 |
| power tail prior 80pct | 150.4 [100.4–330.0] | 2,180 |
| exponential tail prior 80pct | 147.5 [102.1–311.1] | 2,343 |
| detached prior 50pct | 152.8 [102.5–334.3] | 1,728 |

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 142.9 [99.1–290.4] | 1.354 |
| 2 × sampled radius | 145.2 [100.5–298.0] | 1.355 |
| 4 × sampled radius | 147.3 [101.0–303.7] | 1.351 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
