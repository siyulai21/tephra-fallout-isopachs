# Publish and revise a case

The case home is the reading copy. It shows the five principal maps, volumes, validation and material limitations. The audit owns source decisions. The result directories own the fitted fields, volume draws, geographic tests and diagnostics. Cite those files from related studies.

## Revising a result

1. Record the previous commit and input identity before publishing the revised case. Link the previous result at that commit. A historical comparison keeps its own input, folds and reported values.
2. Preserve original sources, row decisions, input hashes, failed fits and alternatives that support a scientific claim. Keep an older file locally when an active study or regression test uses it.
3. Store each result once. Check SHA-256 equality before retiring a duplicate and update its readers and manifests to use the canonical file. For a spherical revision, check whether a source audit still reads the old broad control before replacing `bearing/parent-report.json`. The dated archive can hold that original under `audit-inputs/<slug>/results/spherical-mixture/report.json`. Record the reviewed fingerprints for new or replaced evidence in the [publication inventory](../../benchmarks/published-evidence.sha256).
4. Run the numerical checks for the affected families, the case verifier and the Markdown navigation check. For an editorial change, compare the scientific files with the preceding release. For a figure change, replay the fitted values and inspect the maps.

## Checking a map

The case home shows the three global median-thickness maps and the complete regional hull. Confirm the 0.5 cm footprint fits the chart, the near-vent frame matches across families and each principal number integrates the surface shown. Inspect islands, limits, contour labels and detached deposits. Review the thickness–area table and independent volume checks. A closed contour still needs its support audit. [Output definitions](../OUTPUTS.md#read-the-map-and-its-volume)

### Replay a figure

From the repository root, choose a fresh output directory. For example, to render the B-Tm comparison and verify the published figures, run

```sh
python -m benchmarks.deposit_maps --case b-tm --output results/btm-deposits
python -m benchmarks.verify_visuals
```

Before publishing, inspect labels, coastline, observations and extent. Verify the artifact manifests and compare numerical reports, tables and draws with the preceding release. [Bundled coastline source](../../src/empirical_isopach/assets/coastline-source.json)
