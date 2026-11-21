# Pinatubo 1991 Layer C · Bayesian nested isopachs

**7.6 km³ of bulk fallout**, with a conditional 95% interval of **4.7–41.2 km³**. Each sample integrates one complete deposit over the Earth.

The 137 accepted rows include 102 exact thicknesses, 13 lower bounds, 22 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 1.040. 2/4 training-fold fits passed the population check. The full-fit population check retained a warning. Direct spatial integration passed.

At withheld exact sites, 92/102 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 4.8%. The median fraction beyond the farthest exact site is 20.2%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 0.83 [0.57–31.21] |
| 0.01–0.1 | 2.49 [1.66–8.72] |
| 0.1–1 | 3.18 [1.13–5.25] |
| 1–10 | 0.58 [0.48–0.84] |
| 10–100 | 0.20 [0.15–0.27] |
| 100–1000 | 0.00 [0.00–0.00] |
| >1000 | 0.00 [0.00–0.00] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 7.6 [4.7–41.2] | 24,576 |
| weak profile transfer | 8.0 [4.9–158.1] | 7,106 |
| broader peak prior | 7.6 [4.7–41.3] | 24,575 |
| power tail prior 80pct | 7.7 [4.8–110.1] | 20,111 |
| exponential tail prior 80pct | 7.5 [4.7–12.2] | 24,158 |
| detached prior 50pct | 7.6 [4.7–41.3] | 24,576 |

## Compatibility with fitted bounds

Of 35 documented thickness bounds, 1 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 44 | ≤ 1 | 2.45 | 3.42% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 7.0 [4.5–10.3] | 4.192 |
| 2 × sampled radius | 7.4 [4.6–11.5] | 1.442 |
| 4 × sampled radius | 7.5 [4.7–14.0] | 1.130 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
