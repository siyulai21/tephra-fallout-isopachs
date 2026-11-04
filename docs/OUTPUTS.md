# Reading current maps and volumes

[Methods](METHODS.md) · [Current reports](../eruptions/README.md) · [Regional protocol](methods/regional-comparison.md)

The current `compare` workflow shows the GP ensemble and cubic spline side by side. Both maps use the same footprint and colour/contour scale. Contours stop at the observation hull; they are not artificially closed. Triangles mark documented bounds and stars mark the source reference.

Both expected volumes integrate the same hull. The median-surface integral is separate from the median of random volume. Neither is total eruption volume or DRE. Solid volume bars are conditional joint-surface intervals; dotted ranges show fully retuned geographic deletions and are not confidence intervals. A strongly trend-shrunk spline can have a very narrow conditional interval because trend-estimation uncertainty is omitted.

| Current output | Meaning |
|---|---|
| `README.md`, `isopachs.png/.svg`, `diagnostics.png/.svg` | Paired findings, maps and reliability checks |
| `report.json` | All fitted methods, common region, nested validation, numerical checks and provenance |
| `accepted_observations.csv` | Accepted source rows, values and bound semantics |
| `heldout_predictions.csv` | Outer fold, prediction, interval and score for every accepted observation |
| `selection/*.csv` | All candidate parameters, failures and inner fold scores |
| `predictions.csv` | Regenerable grid, excluded from committed reports |

Read `methods.<method>.regional_volume` for expected and median-surface volumes, `joint_volume` for conditional draws, and `validation.methods.<method>` for outer held-out performance and deletion sensitivity. The labels are `stacked` and `spline`; neither is preferred.


## Historical GP workflow

Versions 0.3–0.7 used different domains and report fields. Read the separate [historical output guide](../archive/reference/historical-gp-outputs.md); those conventions do not describe current regional comparisons.
