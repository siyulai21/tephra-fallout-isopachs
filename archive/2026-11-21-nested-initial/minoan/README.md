# Minoan · Bayesian nested isopachs

**38.3 km³ of bulk fallout**, with a conditional 95% interval of **26.0–111.3 km³**. Each sample integrates one complete deposit over the Earth.

The 101 accepted rows include 92 exact thicknesses, 9 lower bounds, 0 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 1.535. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 85/92 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 6.5%. The median fraction beyond the farthest exact site is 5.8%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 1.43 [0.44–45.06] |
| 0.01–0.1 | 3.50 [1.86–36.00] |
| 0.1–1 | 9.52 [5.87–19.08] |
| 1–10 | 18.17 [12.04–32.10] |
| 10–100 | 3.17 [1.18–6.72] |
| 100–1000 | 0.00 [0.00–3.96] |
| >1000 | 0.00 [0.00–0.12] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 38.3 [26.0–111.3] | 3,072 |
| weak profile transfer | 35.6 [25.8–70.9] | 79 |
| broader peak prior | 38.6 [26.0–110.1] | 3,041 |
| power tail prior 80pct | 38.3 [25.8–128.7] | 2,124 |
| exponential tail prior 80pct | 38.5 [26.4–98.2] | 2,409 |
| detached prior 50pct | 38.0 [25.9–117.6] | 1,880 |

## Compatibility with fitted bounds

Of 9 documented thickness bounds, 0 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 36.0 [24.8–67.3] | 1.537 |
| 2 × sampled radius | 37.0 [25.5–72.1] | 1.535 |
| 4 × sampled radius | 37.8 [25.8–83.8] | 1.535 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
