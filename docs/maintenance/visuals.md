# Reproduce and check a map

Each case opens with four maps of pointwise median thickness. All panels share the same contours, colour scale and geographic frame for that eruption. The full observation hull and every accepted site are in view. The regional fields end at the hull boundary. For the directional ensemble and spherical mixture, the whole-Earth maps show where contours continue beyond the comparison frame.

## What to check

- Read the colour key on the figure. Values below its lowest tick share a shade. The scale on a whole-Earth map can differ from the case comparison.
- Check the bounds, projected positions and clipping against the recorded map design. Inspect the outer contour and any projection-rim warning. The volume calculation uses its stated integration domain.
- Compare the plotted values with the saved fit. The four-panel PNG records source hashes, coordinates, contour levels and five reference predictions per method. The regional replay uses the selected spline settings and GP weights. Both global maps use saved posterior fields. The single-structure directional reference plots geometric mean thickness.
- Keep held-out observation coverage, numerical convergence and the local support screen visible. Each answers a different question about an extrapolated contour. [Support protocol](../methods/supported-isopachs.md)

## Reproduce

From the repository root, choose a fresh output directory. For example, to render the B-Tm comparison and verify the published figures, run

```sh
python -m benchmarks.refresh_figures --case b-tm --scopes comparison --output results/btm-figures
python -m benchmarks.verify_visuals
```

Before publishing, inspect labels, coastline, observations and extent. Verify the artifact manifests and compare numerical reports, tables and draws with the preceding release. [Bundled coastline source](../../src/empirical_isopach/assets/coastline-source.json) · [Publication policy](README.md)
