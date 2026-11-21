# B-Tm · Bayesian nested isopachs

**78.1 km³ of bulk fallout**, with a conditional 95% interval of **53.5–1,049.4 km³**. Each sample integrates one complete deposit over the Earth.

The 71 accepted rows include 64 exact thicknesses, 3 lower bounds, 4 upper bounds and 0 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

The matched geographic score is 1.574. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 60/64 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 6.0%. The median fraction beyond the farthest exact site is 14.3%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.

| Thickness slice (cm) | Median [conditional 95%], km³ |
|---|---:|
| 0–0.01 | 1.34 [0.72–51.01] |
| 0.01–0.1 | 5.85 [3.36–415.08] |
| 0.1–1 | 22.32 [14.49–471.05] |
| 1–10 | 37.32 [24.33–209.83] |
| 10–100 | 7.93 [1.82–35.16] |
| 100–1000 | 0.00 [0.00–1.10] |
| >1000 | 0.00 [0.00–0.00] |

## Prior sensitivity

| Assumption | Median [conditional 95%], km³ | Effective draws |
|---|---:|---:|
| baseline | 78.1 [53.5–1,049.4] | 3,072 |
| weak profile transfer | 78.0 [54.8–1,224.8] | 727 |
| broader peak prior | 78.1 [53.5–1,071.4] | 3,065 |
| power tail prior 80pct | 81.4 [53.7–1,803.8] | 2,301 |
| exponential tail prior 80pct | 77.8 [53.1–704.6] | 2,981 |
| detached prior 50pct | 80.3 [53.7–1,045.8] | 2,246 |

## Compatibility with fitted bounds

Of 7 documented thickness bounds, 2 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Lake Sihailongwan | ≤ 0.06 | 1.09 | 0.64% |
| Quanyang peat | ≤ 0.64 | 8.72 | 0.89% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 77.0 [51.9–554.8] | 1.570 |
| 2 × sampled radius | 77.8 [52.9–616.2] | 1.573 |
| 4 × sampled radius | 78.0 [53.3–716.8] | 1.574 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)
