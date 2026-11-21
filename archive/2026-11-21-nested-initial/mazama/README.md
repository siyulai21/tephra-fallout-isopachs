# Mazama · Bayesian nested isopachs

**350.2 km³ of bulk fallout**, with a conditional 95% interval of **151.2–1,276.5 km³**. Each sample integrates one complete deposit over the Earth.

The 45 accepted rows include 45 exact thicknesses, 0 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 1.467. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 41/45 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 6.2%. The median fraction beyond the farthest exact site is 6.1%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 1.67 [0.60–51.01] |
| 0.01–0.1 | 8.08 [3.16–181.76] |
| 0.1–1 | 40.09 [17.20–246.09] |
| 1–10 | 129.80 [60.66–467.18] |
| 10–100 | 146.99 [48.62–377.43] |
| 100–1000 | 8.15 [0.00–79.21] |
| >1000 | 0.00 [0.00–0.85] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 350.2 [151.2–1,276.5] | 12,288 |
| weak profile transfer | 362.7 [149.0–1,092.1] | 1,095 |
| broader peak prior | 350.1 [151.5–1,283.0] | 11,631 |
| power tail prior 80pct | 352.1 [150.2–1,611.8] | 7,866 |
| exponential tail prior 80pct | 350.2 [152.0–1,136.0] | 10,973 |
| detached prior 50pct | 350.6 [151.6–1,276.5] | 11,850 |

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 341.7 [148.5–1,023.2] | 1.464 |
| 2 × sampled radius | 345.2 [150.1–1,081.7] | 1.466 |
| 4 × sampled radius | 348.0 [150.6–1,112.6] | 1.467 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
