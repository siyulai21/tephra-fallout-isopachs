# Changes

## 0.7.0 — Complete-Earth fallout volumes

- Replace the arbitrary-region headline with an integral over the physical WGS84 Earth, using the exact ellipsoidal LAEA domain.
- Add positive transformed-radial/angular quadrature, two-refinement and phase checks, distance-based tail accounting, and alternate-family/deletion sensitivities.
- Keep fitted models and held-out predictions unchanged; retain regional results explicitly as diagnostics, with no relabelling of their intervals as global uncertainty.
- Publish both methods for all six current eruptions, 24 geographic deletion fits, 30 global known-truth trials and independent geometric/analytic tests.

## 0.6.1 — Reviewed YTT principal dataset

- Include qualified Maninjau primary beds and source-verified U1445B (8 cm), for 46 principal observations.
- Record a decision for all 88 audited records and keep U1448B as an explicit inclusion sensitivity.
- Rebuild inputs from versioned rules without changing the original workbook or inventing uncertainty.
- Separate principal reports, four current sensitivities and archived 43/45-record analyses.
- Recalculate geographic holdouts, maps and regional volumes with unchanged model code.

## 0.6.0 — Combined Minoan fallout and repository cleanup

- Apply one total-fallout field to all Minoan fans using a source audit and explicit data sensitivities.
- Add lower-bound likelihoods consistently to fitting, selection, stacking and validation.
- Preserve exact-measurement coverage denominators and unchanged existing-data predictions.
- Remove the atlas command, HTML/CSS report generator and remaining browser workflow.
- Consolidate scientific results under `reports/` and raw Minoan sources under `data/sources/`.
- Keep the established method as default; report the ensemble comparison and unresolved tails.

## 0.3.0 — Upload workspace and deposit atlas

- Preserve the existing fit and candidate selection.
- Catalogue original CSVs under `data/eruptions/`; archive historical maps without changing bytes.
- Add a local CSV-upload workspace, batch atlas and single-deposit reports.
- Resolve closed contours by domain expansion; subtract holes and flag unresolved levels.
- Refine regional volumes and compare regions/families without claiming total erupted volume or DRE.
- Export maps, GIS contour lines, numerical tables and short reports.
- Retain original import paths, CSV commands and optional spline GUI.

## 0.2.0 — Statistical treatment of unobserved regions

Directional trend plus conditional spatial model, strict observation semantics, exclusion of pseudo-points, spatial selection, independent soft constraints, a normalized spline-prior alternative and diagnostic benchmarks.
