# Publish and revise a case

The case home is the reading copy. It shows the four maps, volumes, validation and material limitations. The audit owns source decisions. The result directories own the fitted fields, volume draws, geographic tests and diagnostics. Cite those files from related studies.

## Revising a result

1. Record the previous commit and input identity before publishing the revised case. Link the previous result at that commit. A historical comparison keeps its own input, folds and reported values.
2. Preserve original sources, row decisions, input hashes, failed fits and alternatives that support a scientific claim. Keep an older file locally when an active study or regression test uses it.
3. Store each result once. Check SHA-256 equality before retiring a duplicate and update its readers and manifests to use the canonical file. For a spherical revision, check whether a source audit still reads the old broad control before replacing `bearing/parent-report.json`. The dated archive can hold that original under `audit-inputs/<slug>/results/spherical-mixture/report.json`. Record the reviewed fingerprints for new or replaced evidence in the [publication inventory](../../benchmarks/published-evidence.sha256).
4. Run the numerical checks for the affected families, the case verifier and the Markdown navigation check. For an editorial change, compare the scientific files with the preceding release. For a figure change, replay the fitted values and inspect the maps.

## Checking a map

Each case opens with four maps of pointwise median thickness. All panels share the same contours, colour scale and geographic frame for that eruption. The full observation hull and every accepted site are in view. The regional fields end at the hull boundary. For the directional ensemble and spherical mixture, the whole-Earth maps show where contours continue beyond the comparison frame.

### What to check

- Read the colour key on the figure. Values below its lowest tick share a shade. The scale on a whole-Earth map can differ from the case comparison.
- Check the bounds, projected positions and clipping against the recorded map design. Inspect the outer contour and any projection-rim warning. The volume calculation uses its stated integration domain.
- Compare the plotted values with the saved fit. The four-panel PNG records source hashes, coordinates, contour levels and five reference predictions per method. The regional replay uses the selected spline settings and GP weights. The directional panel replays the quantitative uncertain-fan-direction packet and records its median-map volume. The spherical panel applies the principal bearing continuation to its own saved posterior. Replay its recorded number of draws per chain and verify both the retained area by compass bearing and the integrated mass. The single-structure directional reference plots geometric mean thickness.
- Keep held-out observation coverage, numerical convergence and the local support screen visible. Each answers a different question about an extrapolated contour. [Support protocol](../methods/supported-isopachs.md)

### Replay a figure

From the repository root, choose a fresh output directory. For example, to render the B-Tm comparison and verify the published figures, run

```sh
python -m benchmarks.refresh_figures --case b-tm --scopes comparison --output results/btm-figures
python -m benchmarks.verify_visuals
```

Before publishing, inspect labels, coastline, observations and extent. Verify the artifact manifests and compare numerical reports, tables and draws with the preceding release. [Bundled coastline source](../../src/empirical_isopach/assets/coastline-source.json)
