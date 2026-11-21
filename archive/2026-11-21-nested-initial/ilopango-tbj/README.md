# Ilopango TBJ (Unit G) · Bayesian nested isopachs

**1,000.6 km³ of bulk fallout**, with a conditional 95% interval of **46.3–2,384.3 km³**. Each sample integrates one complete deposit over the Earth.

The 56 accepted rows include 52 exact thicknesses, 4 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 1.088. 2/4 training-fold fits passed the population check. The full-fit population check retained a warning. Direct spatial integration passed.

At withheld exact sites, 50/52 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 0.2%. The median fraction beyond the farthest exact site is 95.8%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 51.01 [0.28–51.01] |
| 0.01–0.1 | 449.06 [1.42–459.06] |
| 0.1–1 | 412.81 [6.95–1,654.38] |
| 1–10 | 71.50 [21.38–197.53] |
| 10–100 | 12.93 [7.88–21.53] |
| 100–1000 | 2.00 [0.35–3.67] |
| >1000 | 0.00 [0.00–0.87] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 1,000.6 [46.3–2,384.3] | 24,576 |
| weak profile transfer | 1,135.5 [137.9–2,306.3] | 5,390 |
| broader peak prior | 1,027.4 [46.7–2,330.5] | 23,794 |
| power tail prior 80pct | 1,175.9 [50.4–2,479.1] | 20,736 |
| exponential tail prior 80pct | 65.8 [44.5–2,050.5] | 16,030 |
| detached prior 50pct | 1,000.2 [46.3–2,384.3] | 24,528 |

## Compatibility with fitted bounds

Of 4 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 74.4 [45.3–215.3] | 1.089 |
| 2 × sampled radius | 108.1 [46.1–321.9] | 1.088 |
| 4 × sampled radius | 162.0 [46.2–539.4] | 1.088 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
