# B-Tm field audit and source comparisons

The accepted table contains **64 measured thicknesses, three lower bounds and one upper bound**. The regional estimate over the 549,464 km² observation hull is 24.3 km³ for the GP and 28.5 km³ for the spline. Their conditional 95% intervals are 15.7–37.1 and 15.5–66.2 km³, respectively. [Source decisions](../../../../data/audits/B-Tm/README.md) · [Matched regional calculations](matched-regional.csv) · [Samjiyon field reports](../../../../data/audits/B-Tm/samjiyon.md)

These completed source refits use the same six directional structures and weights calibrated without B-Tm. The central estimate integrates the pointwise median isopach map over Earth. Conditional 95% intervals come from integrating whole simulated deposits. The upper tail is sensitive to deposition in unmeasured directions. Volumes are bulk cubic kilometres of combined airborne fallout from both phases.

| Source reading | Median-map integral | Median of whole-deposit volumes | Conditional 95% interval |
|---|---:|---:|---:|
| [7 m at Samjiyon, entirely primary Millennium fallout](samjiyon_scalar/result.json) | 63.9 | 84.8 | 37.4–1,070.3 |
| [1.6 m around Samjiyon, entirely primary Millennium fallout](samjiyon_160/result.json) | 57.3 | 77.3 | 34.5–1,132.7 |
| [Twelve site ranges at their low endpoints](range_low/result.json) | 50.8 | 67.2 | 33.2–991.1 |
| [The same ranges at their high endpoints](range_high/result.json) | 59.2 | 79.1 | 38.8–1,016.5 |
| [Two disputed proximal stacks assigned entirely to Millennium fallout](correlations/result.json) | 46.8 | 64.4 | 29.7–1,222.1 |

Each Samjiyon row assumes that the workbook locates the exposure to which its source refers. The two accounts may describe different exposures. Their intervals condition on those assumptions. This evidence release precedes the complete four-method case update and the one-sided Samjiyon comparisons. [Previous case and maps](https://github.com/siyulai21/tephra-fallout-isopachs/tree/2a52e062b7db84699d5229a25b01282c10301b3e/eruptions/b-tm)
