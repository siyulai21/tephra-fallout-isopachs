# Mazama · Bayesian nested whole-area sensitivity

The median-thickness field integrates to **105 km³** over Earth. Its conditional 95% interval is **86.2–267.3 km³**. [Map integral and numerical check](median-field.json)

Continuing the fitted slopes freely beyond every sampled isopach rank gives a conditional 95% interval of 99.6–513.7 km³. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Every accepted training site has exactly the same predicted thickness in these continuations.

The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is 82.2% (conditional 95% 66.2%–90.4%). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.

The 300 accepted rows include 175 exact thicknesses, 0 lower bounds, 96 upper bounds and 29 two-sided intervals.

![Median isopachs and uncertain total area](isopachs.png)

Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)

## Validation

Accepted CSV row 261 (Fish Lake Valley, Furnace Creek hand auger) sets the last sampled isopach rank in 84.5% of complete draws and 81.9% among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes that measured thinning ends. It does not measure the effect of deleting a site. [Attribution to every measured or bounded site](edge-sites.csv)

The independent populations’ upper 95% endpoints ranged from 250.4 to 275.7 km³. The unmodified thinning profile passed and the continued global volume passed. [Population estimates and refinement history](full.json)

The matched geographic score is 1.257. 4/4 training-fold fits passed the population check. The full-fit population check passed. Direct spatial integration passed.

At withheld exact sites, 156/175 measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is 8.2%. The median fraction beyond the farthest exact site is 26.7%. Geographic classes describe sampling coverage.

No complete contour band passed the shared support screen. The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.

## What contributes to volume

A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Each interval describes uncertainty in its own slice.

| Thickness slice (cm) | Conditional 95% volume interval (km³) |
|---|---:|
| 0–0.01 | 0.51–21.32 |
| 0.01–0.1 | 3.81–44.61 |
| 0.1–1 | 27.91–105.32 |
| 1–10 | 29.74–82.97 |
| 10–100 | 8.57–22.08 |
| 100–1000 | 2.49–13.55 |
| >1000 | 0.00–0.00 |

## Prior sensitivity

| Assumption | Conditional 95% volume interval (km³) | Effective draws |
|---|---:|---:|
| baseline | 86.2–267.3 | 49,152 |
| weak profile transfer | 86.9–267.4 | 42,558 |
| broader peak prior | 86.2–272.3 | 49,042 |
| power tail prior 80pct | 86.6–281.4 | 36,503 |
| exponential tail prior 80pct | 85.5–250.3 | 35,785 |
| detached prior 50pct | 84.6–246.2 | 34,382 |
| more distal ash prior | 86.6–267.5 | 46,803 |
| less distal ash prior | 86.1–273.0 | 48,902 |
| distal prior without Santa Maria 1902 | 84.5–251.3 | 33,760 |
| distal prior without Quizapu 1932 | 86.3–269.3 | 48,911 |
| distal prior without Novarupta 1912 | 86.3–269.2 | 48,832 |
| distal prior without Hudson 1991 | 86.4–268.8 | 48,387 |

## Compatibility with fitted bounds

Of 125 documented thickness bounds, 12 had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error.

| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |
|---|---|---:|---:|
| Row 104 | 28–30 | 44.9 | 2.74% |
| Row 264 | 26–27 | 7.72 | 0.63% |
| Row 102 | 12–13 | 5.69 | 2.38% |
| Row 103 | 25–30 | 6.49 | 2.11% |
| Row 19 | 40–72 | 6.1 | 1.55% |
| Row 24 | 63–90 | 4.8 | 0.26% |
| Row 25 | 30–70 | 4.78 | 1.91% |
| Row 36 | 15–30 | 3.24 | 3.39% |
| Row 268 | 26–27 | 2.37 | 0.05% |
| Row 57 | 10–30 | 1.53 | 1.97% |
| Row 56 | 7–9 | 1.51 | 1.92% |
| Row 81 | 0.4–0.5 | 1.83 | 2.64% |

The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model.

## Additional distal thinning

Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.

| Additional range | Conditional 95% volume interval (km³) | Geographic score ↓ |
|---|---:|---:|
| 1 × sampled radius | 86.0–242.3 | 1.265 |
| 2 × sampled radius | 86.1–251.6 | 1.260 |
| 4 × sampled radius | 86.2–259.6 | 1.258 |

The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)

The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.

[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)

[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)
