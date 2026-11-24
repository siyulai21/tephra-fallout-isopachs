# Report figures and offline cartography

[Shared framework](../README.md) · [Output guide](../../../docs/OUTPUTS.md) · [Rendering policy](../../../docs/maintenance/README.md#checking-a-map)

This package reads fitted thickness distributions, integrates their published maps and renders the figures. The fitting code and accepted observations are maintained by their model and data modules.

| File | Responsibility |
|---|---|
| [deposit.py](deposit.py), [deposit_field.py](deposit_field.py), [deposit_integrals.py](deposit_integrals.py) and [deposit_maps.py](deposit_maps.py) | Verified global median surfaces, whole-Earth map integrals, doubling contours, land and lake fills, near-vent detail and thickness–area curves |
| [principal.py](principal.py) | Five-method table of median-map integrals, integrated thickness quartiles and geographic validation |
| [contour_limits.py](contour_limits.py) and [shorelines.py](shorelines.py) | Site-enclosing contour levels, matched local windows and scale-dependent land and lake polygons |
| [thickness_band.py](thickness_band.py) | Pointwise thickness quantiles, local hull and whole-Earth integration, and independent numerical checks |
| [maps.py](maps.py) | Pinned fine coastlines, complete observation frames, aspect-aware layouts and global display-edge checks |
| [style.py](style.py) | Shared colours and typography for report figures |
| [regional.py](regional.py) | Common-scale median fields clipped to the exact observation hull |
| [global_maps.py](global_maps.py) | Wider global continuation, complete local detail, whole-Earth context and latent uncertainty |
| [comparison.py](comparison.py) | Four median isopach maps in a common observation-area frame and the case results table |
| [thickness_band.py](thickness_band.py), [principal.py](principal.py) | Integrate Q25, Q50 and Q75 thickness on the same domain and publish a table tied to one accepted input |
| [summary.py](summary.py) | Separate regional/global volume panels and matched geographic prediction scores |

All three global principal maps show pointwise median thickness. The full extent follows the observation-coverage rule. Each eruption shares one proximal viewport across methods. The earlier single-structure directional study displays geometric-mean thickness. The earlier four-panel comparison uses the saved ensemble median grid, the spherical posterior display draws and the two regional median fields clipped to the observation hull. Each comparison PNG records its inputs, scale and extent in the Description metadata.

Regional and historical diagnostic figures can be replayed without retuning with `python -m benchmarks.refresh_figures --output results/new-figures`. The replay checks regional parameters and integrals against the stored execution; it reads global draws unchanged. Publish only reviewed visual files and their explicit provenance. Never copy the scratch prediction raster over a published scientific table.
