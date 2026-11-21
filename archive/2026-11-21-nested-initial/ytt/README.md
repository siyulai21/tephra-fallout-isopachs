# YTT · Bayesian nested isopachs

**1,763.5 km³ of bulk fallout**, with a conditional 95% interval of **1,289.1–3,047.2 km³**. Each sample integrates one complete deposit over the Earth.

The 46 accepted rows include 46 exact thicknesses, 0 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 1.694. 3/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 42/46 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 16.3%. The median fraction beyond the farthest exact site is 0.7%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 14.91 [9.94–36.01] |
| 0.01–0.1 | 71.52 [50.74–125.92] |
| 0.1–1 | 327.24 [238.09–544.16] |
| 1–10 | 881.83 [691.48–1,707.67] |
| 10–100 | 409.31 [126.78–795.94] |
| 100–1000 | 0.20 [0.00–215.38] |
| >1000 | 0.00 [0.00–1.87] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 1,763.5 [1,289.1–3,047.2] | 3,072 |
| weak profile transfer | 1,744.7 [1,138.2–3,047.2] | 254 |
| broader peak prior | 1,769.2 [1,289.1–3,047.2] | 3,050 |
| power tail prior 80pct | 1,770.3 [1,279.0–3,560.1] | 2,136 |
| exponential tail prior 80pct | 1,759.9 [1,298.9–2,778.8] | 2,930 |
| detached prior 50pct | 1,719.6 [1,285.6–3,560.1] | 1,808 |

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 1,763.3 [1,287.1–3,046.2] | 1.673 |
| 2 × sampled radius | 1,763.4 [1,288.3–3,046.9] | 1.688 |
| 4 × sampled radius | 1,763.5 [1,288.9–3,047.1] | 1.692 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
