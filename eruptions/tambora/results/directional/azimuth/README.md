# Tambora · directional extent

The same 26 accepted observations give a **107.7 km³** median-map integral with fixed fan directions and **95.7 km³** when an unsampled fan can turn. These two fits use identical observations and structure weights. The primary six-structure estimate on the case home uses its original calibration.

Under uncertain azimuth, integrating each complete deposit gives a volume median of **117.5 km³** and a conditional 95% range of **73.2–203.8 km³**. Measured interpolation triangles contribute 11.2% of the median-map volume. The rest relies on continuation.

![Median isopachs and quantitatively bracketed segments](median-isopachs.png)

Solid marks require a triangle whose measured sites bracket the thickness and whose posterior crossing probability reaches 90%. Dashes show the remaining median contour. White circles are thickness observations. An empty sector can contain fallout.

## Reach from the volcano

Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.

| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |
|---|---:|---:|---:|---:|
| N | 296.1 | 282.8 | 36,159 | 35,485 |
| NE | 267.8 | 267.6 | 29,559 | 28,957 |
| E | 279.0 | 273.7 | 30,113 | 28,564 |
| SE | 203.9 | 179.3 | 17,050 | 12,975 |
| S | 253.8 | 208.4 | 24,735 | 18,708 |
| SW | 263.7 | 249.6 | 28,248 | 24,406 |
| W | 447.2 | 447.1 | 58,228 | 57,990 |
| NW | 299.9 | 216.8 | 38,085 | 28,038 |

![Probability that a deposit reaches ten centimetres](exceedance.png)

The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.

## Withheld observations

Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. A dash means no fold passed.

| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing |
|---|---:|---:|---:|---:|---:|---:|
| geographic | 4 | 1.941 | 2.066 | 1 | 2.585 | 2.585 |
| quadrants | 4 | 1.763 | 1.904 | 0 | — | — |

[Predictions for every estimable held-out site](heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.

## Checks and reproducibility

The shared contour audit found eligible bands under none of the screened support policies. Audit status is completed. [All contour decisions and deletion failures](complete-contour-audit.json)

The spatial integration check passed at 8,192 points per scramble. The map used up to 64 posterior draws per chain and structure. Resolved contour areas passed the split-packet check. The joint-volume quantiles passed the split check.

Full-fit sampling warnings occurred for narrow_fans, reference, narrow_missing_fan, possible_missing_fan.

[Numerical report, all sector volumes and sensitivities](report.json) · [Complete fields and per-field volumes](fields.npz) · [Plotted grids](map-grid.npz) · [Fragment decisions](fragment-decisions.csv) · [Bracketed coordinates](bracketed-fragments.csv) · [File hashes](artifacts.json)
