# Can a change in radial thinning improve geographic predictions?

Six radial thinning laws use the same 68 accepted B-Tm observations and the same reference fan geometry. Five have predictions for all four withheld geographic regions. The wider slope prior is a full-data sensitivity. [Accepted observations](../../../../../data/eruptions/B-Tm_reviewed.csv) · [Whole-deposit volumes](volumes/README.md)

| Thinning law | Full-data sampling | Mean held-out score | Geographic folds with warnings |
|---|---|---:|---|
| Published smooth reference | Passed | 1.425 | None |
| One shared break | Passed | 1.425 | None |
| Two shared breaks | Passed | 1.441 | None |
| Exponential kernels | Passed | 1.408 | None |
| Exponential kernels, one break | Passed | 1.423 | None |
| One break, wider slope prior | Passed | — | Full fit only |

Each region receives equal weight. Lower negative log predictive scores indicate better predictions at withheld measurements and bounds. Adding one break changes the four scores by +0.018, +0.007, -0.030, +0.005. [Individual held-out predictions](heldout.csv) · [Model scores](summary.csv)

The one-break fit places the change at a median radius of 60.6 km (conditional 95% interval 2.9–1,124 km). The radial slope multiplier is 1.24 (0.44–3.58). The interval allows both steeper and gentler thinning.

The smooth reference combines short-range and bearing-dependent components. The break models change the radial slope continuously at one or two distances shared across bearings. Each training fold sets its own prior range from its nearest and farthest positive site radii. The classical segmented-exponential approach uses the square root of enclosed isopach area. These fits use distance from the vent. [Directional equations](../../../../../docs/methods/directional-continuation.md) · [Fierstein and Nathenson (1992)](https://pubs.usgs.gov/publication/70017275)

[Fold membership, attempts and parameter quantiles](fits.json) · [Break-parameter draws](break-draws.npz) · [Checksums and software versions](design.json)

## Reproduce

```sh
for v in smooth one_break two_breaks exponential broken_exponential wide_prior; do
  python -m benchmarks.btm_thinning --work results/btm-break-check --variant "$v"
done
python -m benchmarks.btm_thinning_report --work results/btm-break-check --output results/btm-break-report
```
