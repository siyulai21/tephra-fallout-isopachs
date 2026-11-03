# Historical 0.3 reporting validation

This is an archived record. The atlas and browser application were removed in 0.6; current results live under `reports/`.

The 0.3 reporting workflow was checked on 1–2 November 2026. The statistical
model remains the one documented in [METHODS.md](../METHODS.md); the earlier
research benchmark and its unfavorable results remain in [VALIDATION.md](../validation/initial-benchmark.md).

## Unchanged scientific fit

- `model.py`, `data.py` and `validation.py` are byte-identical to the previous
  committed implementation.
- All four input CSVs and historical PNGs were moved without changing bytes.
- Every fitted model description and nested-validation result in the new atlas
  exactly matches its previous saved `results/validated/` counterpart.
- No invented points, historical wind profiles or new eruption assumptions
  were introduced.

## Numerical reporting

The checked-in atlas uses a 961×961 display grid, with local refinement of
small contours. All resolved contour areas meet the 2% comparison criterion;
the largest remaining change is below 0.2%. Analytic circle and annulus tests
verify area calculation, including hole subtraction. An unclosed level never
receives an area.

| Deposit | Expected regional bulk volume, km³ | Median-surface integral, km³ | Volume outside data hull | Outermost displayed closed contour |
|---|---:|---:|---:|---:|
| Mazama | 171.7 | 159.0 | 23.4% | 0.1 cm |
| Tambora | 136.3 | 122.1 | 27.5% | 0.1 cm |
| Aso-4 | 646.5 | 491.2 | 67.8% | 1 cm |
| B-Tm | 40.1 | 36.3 | 28.3% | 1 cm |

All 24 volume calculations (four deposits × three regions × two families)
passed the 0.5% successive-grid criterion. These finite-region integrals are
not total erupted volumes or DRE. Differences from earlier coarse-grid values
reflect numerical refinement, not a changed fit.

The 0.1 cm contours for Aso-4 and B-Tm extend beyond the default map limit and
are explicitly unresolved. Thicker curves close naturally on the evaluated
surface. Their outer closures remain extrapolations.

## Workflow checks

- The numerical, statistical, upload-preview and local HTTP tests pass.
- An actual browser-selected CSV was previewed and analysed through the local
  workspace; a bundled example was also run end to end.
- HTML reports, map images, data tables, short reports and download links were
  inspected. Static atlas links resolve locally.
- A wheel builds successfully and contains the catalogue, all four CSVs,
  offline coastline asset and browser workspace. Editable installation and
  the command-line entry point were checked.

Conditional volume quantiles remain secondary diagnostics: their grid
convergence, trend uncertainty and model uncertainty are not established.
The GUI does not change the known Aso-4 undercoverage, missing-sector failure,
or unaudited source-provenance limitations.
