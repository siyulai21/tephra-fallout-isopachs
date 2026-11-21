# Sakurajima Satsuma (P14) · Bayesian nested isopachs

**10.5 km³ of bulk fallout**, with a conditional 95% interval of **6.6–709.8 km³**. Each sample integrates one complete deposit over the Earth.

The 57 accepted rows include 50 exact thicknesses, 5 lower bounds, 2 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 0.751. 2/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 48/50 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 1.6%. The median fraction beyond the farthest exact site is 32.3%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 0.18 [0.05–51.01] |
| 0.01–0.1 | 0.59 [0.23–416.89] |
| 0.1–1 | 2.09 [1.10–235.56] |
| 1–10 | 4.62 [3.21–26.08] |
| 10–100 | 2.01 [1.48–2.91] |
| 100–1000 | 0.00 [0.00–0.17] |
| >1000 | 0.00 [0.00–0.00] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 10.5 [6.6–709.8] | 24,576 |
| weak profile transfer | 9.2 [6.6–971.7] | 4,652 |
| broader peak prior | 10.5 [6.6–725.5] | 24,560 |
| power tail prior 80pct | 11.1 [6.5–1,166.6] | 16,054 |
| exponential tail prior 80pct | 10.2 [6.6–288.7] | 20,687 |
| detached prior 50pct | 10.5 [6.6–698.9] | 22,715 |

## Compatibility with fitted bounds

Of 7 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 9.1 [6.4–19.5] | 0.750 |
| 2 × sampled radius | 9.6 [6.5–29.5] | 0.751 |
| 4 × sampled radius | 10.0 [6.5–50.3] | 0.751 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
