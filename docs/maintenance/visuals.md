# Refresh scientific figures without changing the calculation

[Maintenance policy](README.md) · [Figure implementation](../../src/empirical_isopach/visualization/README.md) · [Case catalogue](../../eruptions/README.md)

The 2026-11-17 refresh adds fine offline coastlines and reader-oriented comparisons to all six current cases. It changes display grids, framing and layout. Accepted inputs, source audits, selected model settings, held-out predictions, numeric volume tables, support decisions and saved posterior arrays remain unchanged. Original execution provenance remains in each `report.json`.

## Read the maps

- Regional maps contain every accepted location, the complete hull and the source. The raster is clipped to the exact hull. Display padding does not enlarge the spline fitting domain or regional integration domain. Gaps inside the hull still depend on the model.
- Each global family has a wider overview, a local-detail view covering all observations, and a whole-Earth Mollweide companion. The wider view searches the physical Earth for the displayed **0.1 cm** central-field contour and pads/checks its edges. Long-tailed fields can reach the projection rim; those cases are explicitly marked. Longitude edges join in the whole-Earth view. These are finite-grid display checks, not proofs of contour support or total-volume identification.
- Global colour represents the spherical posterior median or directional geometric mean, as labelled. The uncertainty map shows the ratio of upper to lower latent-thickness quantiles in orders of magnitude. Observation scatter is additional. Every global plotting subset is deterministic and spans the saved chains; it is recorded separately from the posterior used for volume.
- A local detail or source-region zoom can cut a contour by design. Follow the wider overview and whole-Earth companion. The plot window, colour limits and displayed thickness levels never set the volume.
- Global colours saturate below 0.1 cm and above the displayed upper limit. The lowest colour represents all values at or below that display threshold, not an exact 0.1 cm deposit. Published numerical records retain the full continuous fields through their posterior parameters.
- Case overviews keep regional hull volumes and conditional whole-Earth volumes in separate panels. Geographic scores compare identical accepted rows and outer folds. Withheld observation coverage is not coverage of unknown global volume. Numerical warnings remain visible.

The coastline is Natural Earth 1:10-million, bundled with a pinned source commit, public-domain attribution and original/compressed checksums. [Source record](../../src/empirical_isopach/assets/coastline-source.json).

## Reproduce and verify

```sh
python -m benchmarks.refresh_figures --output results/new-figures
python -m pytest -q
python scripts/check_repository_navigation.py
python -m benchmarks.verify_visuals
python -m benchmarks.verify_aso_evidence
python -m benchmarks.verify_directional
```

Use `--case minoan` for one case, or `--scopes regional` to render only the regional figures. Always choose a fresh ignored destination. The replay reconstructs the recorded regional candidate settings and GP weights without repeating selection, checks model descriptions and hull integrals, and reads the original global posterior arrays. It does not sample, retune or change report values.

Before publication, inspect all map frames and figure labels. Copy only the reviewed images and new `map-display.json` / `figure-replay.json` records; retain numerical tables unchanged. Update current artifact manifests explicitly. The [revision ledger](visuals-2026-11-17.json) records before/after presentation hashes, unchanged scientific files and the original Git snapshot. Frozen historical figures and source-study records retain their original bytes. The Aso evidence verifier permits only exact, named presentation revisions and still rejects numerical-file changes.
