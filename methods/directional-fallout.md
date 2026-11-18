# Directional global fallout from measured isopach fragments

The central directional estimate integrates the pointwise median thickness from six plausible fan structures over the Earth. Each report also gives a conditional interval from complete simulated deposits and a measured-fragment map. [Ensemble method and equations](../docs/methods/directional-ensemble.md) · [Case estimates and stress tests](../comparisons/directional-ensemble/README.md)

## Underlying model and single-structure studies

[Method catalogue](README.md) · [Spherical-mixture peer](spherical-mixture.md) · [Implementation map](../src/empirical_isopach/models/directional/README.md)

**Public command: `global --geometry directional`; also included by `compare`.** Output folder: `directional/`.

This method uses the same observations and bounds as the regional methods. Measured thickness constrains local triangles. Their edges can carry part of an isopach even when the rest of the contour crosses a sampling gap. Each draw integrates all thickness levels inside these triangles.

Across a gap the model uses a positive, source-centred thinning law. Long, closely sampled directions can support a broader fan. An isolated distal measurement can lengthen a narrower range of bearings. In an empty direction the reference reverts to a common, estimated proximal scale. A separate refit allows a previously unseen fan. An additional proximal cap quantifies how much the total can change inside the nearest measurement radius while giving exactly the same thickness at every observed site. None of the fits assigns a zero thickness to missing ground.

The published report shows the bulk volume and its conditional interval, the part inside the triangles, the volume by compass direction, the effects of 12 alternative assumptions and the results of geographic deletion. The same rules apply to every eruption, including deposits with multiple known fans. An entire unseen deposit component remains unidentified by the data. [Model, priors and numerical checks](../docs/methods/directional-continuation.md) · [Six-eruption comparison and known-volume tests](../comparisons/directional/README.md) · [Other global family](spherical-mixture.md).
