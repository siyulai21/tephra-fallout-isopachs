# How the four models fit fallout

Each model starts from the accepted measurements, documented thickness bounds and vent location. Each fits a thickness field and integrates it over its stated area. The case reports keep the four estimates separate.

| Method | Area | How thickness and volume are estimated |
|---|---|---|
| [Directional ensemble](directional-fallout.md) | Whole Earth | Interpolates locally in measurement triangles and fits six ways for ash to thin across gaps. Weights learned from other eruptions combine the fields. Its central volume integrates their pointwise median map. |
| [Spherical mixture](spherical-mixture.md) | Whole Earth | Fits a mixture of spatial fans whose masses sum to the whole-deposit volume. The posterior gives a median and a conditional interval for that volume. |
| [GP ensemble](gaussian-process.md) | Observation hull | Combines spatial Gaussian-process fits by geographic predictive performance. Each volume draw uses one component over the whole hull. |
| [Trend-regularized cubic spline](cubic-spline.md) | The same observation hull | Fits a cubic basis with smoothness and trend penalties. Geographic training folds select the settings without hand-placed pseudo-observations. |

## Two ensembles answer different questions

The **regional GP ensemble** weights candidates using the eruption’s training data. The **global directional ensemble** fits three fan half-widths, each with and without an additional fan in the largest distal sampling gap. Its predictive weights come from independent calibration eruptions, with the target excluded. Each draw for global volume takes one structure across the Earth. The median of those volume draws can differ from the integral of the pointwise median map.

The directional ensemble supersedes the single-structure headline within its family. The earlier fit still tests assumptions outside the six structures, including a proximal ash cap that vanishes at every measurement. [Directional ensemble protocol](../docs/methods/directional-ensemble.md) · [Single-structure sensitivity study](../comparisons/directional/README.md)

All four use documented bounds as likelihood events. The shared support audit identifies interpolation patches, unsupported gaps and rejected contours. Regional intervals condition on fitted settings. Global intervals condition on the model’s assumptions about unmeasured deposits. Held-out scores measure predictive skill at sampled thicknesses. [Matched comparisons and stress tests](../comparisons/README.md)
