# Model response to the distal field evidence

The comparisons below use median-map volumes and local median thicknesses from the saved fits. The GP and cubic spline share the fixed hull of 26 historical measurements in the source comparison. Their case-home maps cover the larger current observation hull. The other three families integrate a conditional continuation over Earth. Each volume interval comes from the corresponding family’s draws. [Field evidence and original references](../../../../data/audits/Mazama/distal-review/FIELD_EVIDENCE.md) · [Principal maps, current-hull volumes and intervals](../../README.md)

## Volumes under the tested readings

All entries are in km³. Each row uses one method and one integration domain. A dagger marks a retained fit, draw or integration warning. The earlier 244 rows are refitted with the source-comparison procedure. The archived standalone 244-position calculation has its own geographic tuning and observation hull. [Original 244-position results](../../../../archive/2026-11-24-mazama-distal-facies/README.md)

| Method | Revised map (conditional 95%) | Earlier 244 rows | Alternate field bounds | Restore thick readings | Omit revised ceilings | Half capture | Omit all ceilings |
|---|---:|---:|---:|---:|---:|---:|---:|
| GP ensemble | 91.2 (60.1–202) | 167 | 91.1 | 162 | 137 | 99.2 | 222 |
| Cubic spline | 194 (147–1.59e+04) † | 227 † | 195 † | 345 † | 208 | 195 † | 396 |
| Directional ensemble | 214 (154–642) † | 332 † | 215 † | 323 † | 248 † | 220 | 333 |
| Spherical mixture | 241 (154–1.58e+03) | 352 † | 242 | 371 | 258 | 239 | 329 |
| Bayesian nested corridors | 175 (106–605) | 275 | 177 | 208 | 183 | 180 | 223 |

Restoring the thick readings reinstates the previous interpretations at nine archives, including Windy and Thunder lakes in British Columbia and Round Lake in Washington. The field alternative varies Johnson, Kilpoola, Foy and three Greater Victoria core brackets together. Half capture doubles the revised package ceilings as a stress test. Each scenario changes a group of evidence decisions, so the differences cannot apportion causality to one site. [All unrounded volumes, conditional intervals and flags](source-sensitivity-volumes.csv) · [Scenario definitions and input hashes](sensitivities/scenarios.json)

## Modelled thickness at the reviewed locations

Each cell gives the **earlier 244-row refit → revised principal** median thickness in cm. These are fitted latent thicknesses. Laboratory identities, enclosing-section bounds and geological observations are recorded separately. A dash indicates an unavailable prediction on the regional domain. [Site intervals and all comparisons](local-thickness-comparison.csv) · [Observed beds, bounds and withheld readings](../../../../data/audits/Mazama/distal-review/locality-facies-crosswalk.csv)

| Site | GP ensemble | Cubic spline | Directional ensemble | Spherical mixture | Bayesian nested corridors |
|---|---:|---:|---:|---:|---:|
| Windy Lake, BC | 11.9 → 0.687 | 20 → 0.131 | 27.3 → 1.02 | 13.2 → 3.46 | 3.84 → 0.984 |
| Thunder Lake, BC | 4.53 → 0.279 | 15.5 → 0.0892 | 3.75 → 0.365 | 1.28 → 0.424 | 2.38 → 0.244 |
| Johnson Lake | 6.33 → 0.265 | 5.81 → 0.0263 | 6.32 → 0.701 | 3.46 → 0.931 | 2.57 → 0.437 |
| Sheep Mountain Bog | 6.7 → 1.96 | 9.34 → 2.08 | 3.21 → 1.43 | 8.1 → 5.22 | 2.19 → 1 |
| Foy Lake core A | 8.37 → 1.53 | 4.31 → 0.726 | 5.23 → 1.28 | 14 → 5.28 | 2.6 → 1.28 |
| Lyman Lake | 8.25 → 2.75 | 9.44 → 5.2 | 6.15 → 2.47 | 9.62 → 4.84 | 5.79 → 1.84 |
| Yellowstone Lake YL16-5A | 0.738 → 0.559 | 0.489 → 0.497 | 0.609 → 0.553 | 0.7 → 0.519 | 1.74 → 0.984 |

## When the surrounding geographic region is withheld

The two regional methods predict each observation using other geographic folds. Values below are predictive medians in cm. For a ceiling, the probability column gives the fitted probability of an observation at or below that ceiling; the measurement rows have no one-sided event. A predictive median above a conditional field ceiling shows a conflict under the local-retention assumption. The next columns give the probability assigned to the observed bound. [Withheld intervals, folds and scores for every observation](heldout-source-observations.csv)

| Site | Observed point or bound (cm) | GP median | GP bound probability | Spline median | Spline bound probability |
|---|---:|---:|---:|---:|---:|
| Windy Lake, BC | ≤86 | 0.704 | 98.9% | 1.06 | 84.1% |
| Thunder Lake, BC | ≤32 | 0.183 | 99.3% | 0.232 | 85.9% |
| Johnson Lake | ≤156 | 0.312 | 99.8% | 0.321 | 89.5% |
| Sheep Mountain Bog | ≤42 | 12.9 | 75.6% | 8.08e+03 | 7.9% |
| Foy Lake core A | ≤8 | 8.47 | 48.6% | 40.9 | 24.0% |
| Lyman Lake | 4 | 4.83 | — | 4.46 | — |
| Yellowstone Lake YL16-5A | 0.5 | 7.06 | — | 0.0183 | — |

The geographic folds assess prediction away from training data. The contour screen tests which bands have observational support, including unsupported interior holes. The complete-Earth estimates condition on the family, continuation, retained evidence and numerical status. [Geographically withheld sites](heldout-source-observations.csv) · [Probabilities at held-out bounds](heldout-bound-probabilities.csv) · [Regional contour support](../support/README.md) · [Global support decisions by family](../../../../comparisons/support/README.md) · [Numerical checks](publication-checks.json)
