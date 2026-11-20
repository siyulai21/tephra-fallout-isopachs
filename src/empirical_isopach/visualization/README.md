# Report figures and offline cartography

[Shared framework](../README.md) · [Output guide](../../../docs/OUTPUTS.md) · [Rendering policy](../../../docs/maintenance/README.md#checking-a-map)

This package presents recorded fits. It does not choose models, alter observations, or integrate global volume.

| File | Responsibility |
|---|---|
| [maps.py](maps.py) | Pinned fine coastlines, complete observation frames, aspect-aware layouts and global display-edge checks |
| [regional.py](regional.py) | Common-scale median fields clipped to the exact observation hull |
| [global_maps.py](global_maps.py) | Wider global continuation, complete local detail, whole-Earth context and latent uncertainty |
| [comparison.py](comparison.py) | Four median isopach maps in a common observation-area frame and the case results table |
| [summary.py](summary.py) | Separate regional/global volume panels and matched geographic prediction scores |

The current directional ensemble and spherical-mixture maps show pointwise median thickness. The earlier single-structure directional study displays geometric-mean thickness. The four-panel comparison uses the saved ensemble median grid, the spherical posterior display draws and the two regional median fields clipped to the observation hull. Each comparison PNG records its inputs, scale and extent in the Description metadata.

Current case figures can be replayed without retuning with `python -m benchmarks.refresh_figures --output results/new-figures`. The replay checks regional parameters and integrals against the stored execution; it reads global draws unchanged. Publish only reviewed visual files and their explicit provenance. Never copy the scratch prediction raster over a published scientific table.
