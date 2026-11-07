# Reading current maps and volumes

[Methods](METHODS.md) · [Current reports](../eruptions/README.md) · [Regional protocol](methods/regional-comparison.md)

The current `compare` workflow shows regional GP and spline results side by side, followed by a separate `global/` companion. Regional maps use the same footprint and colour/contour scale. Their contours stop at the observation hull; they are not artificially closed. Triangles mark documented bounds and stars mark the source reference.

Read `support/README.md` first when deciding which parts to use for volume or extrapolation. It distinguishes local interpolation patches from unsampled holes, tests complete contours, and reports unresolved cases explicitly. A hull or closed contour is not automatically supported evidence.

Both expected volumes integrate the same hull. The median-surface integral is separate from the median of random volume. Neither is total eruption volume or DRE. Solid volume bars are conditional joint-surface intervals; dotted ranges show fully retuned geographic deletions and are not confidence intervals. A strongly trend-shrunk spline can have a very narrow conditional interval because trend-estimation uncertainty is omitted.

| Current output | Meaning |
|---|---|
| `README.md`, `isopachs.png/.svg`, `diagnostics.png/.svg` | Paired findings, maps and reliability checks |
| `report.json` | All fitted methods, common region, nested validation, numerical checks and provenance |
| `accepted_observations.csv` | Accepted source rows, values and bound semantics |
| `heldout_predictions.csv` | Outer fold, prediction, interval and score for every accepted observation |
| `selection/*.csv` | All candidate parameters, failures and inner fold scores |
| `predictions.csv` | Regenerable grid, excluded from committed reports |
| `support/README.md`, `support/support.png` | Local footprints, partial volumes and contour eligibility for both methods |
| `support/contour_audit.csv` | Each candidate level, policy, stability check and rejection reason |
| `support/report.json` | Three support policies, deletion sensitivity, training-only support strata, band checks and conditional extrapolation scenarios |
| `support/support_geometry.json` | Footprint polygons in source-centred projected kilometres; not longitude/latitude |
| `global/README.md`, `global/field.png`, `global/volume.png` | Conditional whole-Earth estimates, field uncertainty, assumption sensitivity and unobserved mass |
| `global/report.json`, `global/estimates.csv` | All global priors, numerical attempts, diagnostics, posterior quantiles and provenance |
| `global/posterior.npz` | Baseline whole-field parameter draws, retaining independent chain identity |
| `global/volume_draws.csv.gz` | Global volume draws for all assumptions and chains; no observation noise added |
| `global/heldout_predictions.csv` | Global-model predictions using the same geographic outer folds as the regional peers |
| `global/source.png`, `global/report.json` → `source_checks` | Source-region contours, vent thickness, explicit shape constraints and numerical normalization checks when supplied by the reviewed model |

Read `methods.<method>.regional_volume` for expected and median-surface volumes, `joint_volume` for conditional draws, and `validation.methods.<method>` for outer held-out performance and deletion sensitivity. The labels are `stacked` and `spline`; neither is preferred.

In the support report, `methods.<method>.footprint_volumes` contains partial volumes on the shared local triangles. `contours.policies` records eligible bands and continuation status. `preferred_global_km3` remains null: scenario extrapolations are not validated global estimates. The three policy ranges and geographic-deletion ranges are not confidence intervals. See the [support protocol](methods/supported-isopachs.md).

The global report is separate: `fits.baseline.volume` contains the global median and conditional 95% credible interval. Each alternative in `fits` has its own posterior; `sensitivity_envelope_km3` is not a 95% interval. `mass_partition` describes sampling geometry, not geological confidence. `diagnostics.passed` concerns numerical mixing and, for source-focused fits, normalization accuracy; it does not establish model correctness or global identifiability. `normalization_refinements` retains lower-resolution fits when a full posterior refit was necessary. The old support gate does not suppress these conditional global estimates. See the [global protocol](methods/probabilistic-global.md) and [source-focused extension](methods/source-anchored-fans.md).

Mass fractions use finite positive quadrature. A very small part can receive no quadrature points and be reported as zero even though the fitted field is everywhere positive. Read the fraction-refinement error alongside small proximal fractions; zero is not evidence of absent ash. This partition approximation does not determine the total volume.


## Historical GP workflow

Versions 0.3–0.7 used different domains and report fields. Read the separate [historical output guide](../archive/reference/historical-gp-outputs.md); those conventions do not describe current regional comparisons.
