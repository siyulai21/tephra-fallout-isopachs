# Report figures and offline cartography

[Shared framework](../README.md) · [Output guide](../../../docs/OUTPUTS.md) · [Rendering policy](../../../docs/maintenance/visuals.md)

This package presents recorded fits. It does not choose models, alter observations, or integrate global volume.

| File | Responsibility |
|---|---|
| [maps.py](maps.py) | Pinned fine coastlines, complete observation frames, aspect-aware layouts and global display-edge checks |
| [regional.py](regional.py) | Common-scale median fields clipped to the exact observation hull |
| [global_maps.py](global_maps.py) | Wider global continuation, complete local detail, whole-Earth context and latent uncertainty |
| [summary.py](summary.py) | Separate regional/global volume panels and matched geographic prediction scores |

Both global families retain their original plotted statistic: posterior median for the spherical mixture, geometric-mean thickness for directional continuation. Their global panels are model summaries, not evidence that unsampled areas contain the displayed thickness. See each figure's `map-display.json` for its grid, field subset, bounds and coastline hash.

Current case figures can be replayed without retuning with `python -m benchmarks.refresh_figures --output results/new-figures`. The replay checks regional parameters and integrals against the stored execution; it reads global draws unchanged. Publish only reviewed visual files and their explicit provenance. Never copy the scratch prediction raster over a published scientific table.
