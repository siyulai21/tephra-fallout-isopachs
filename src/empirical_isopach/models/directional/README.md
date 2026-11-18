# Directional continuation: implementation map

[Family catalogue](../README.md) · [Plain-language guide](../../../../methods/directional-fallout.md) · [Equations](../../../../docs/methods/directional-continuation.md) · [Published evidence](../../../../comparisons/directional/README.md)

This is an independent global family. It fits latent log thickness on retained measurement triangles and a positive source-centred continuation across the rest of Earth. It does not call the GP or cubic-spline fitter, and it does not use another model's contours as observations.

| Module | Responsibility |
|---|---|
| [geometry.py](geometry.py) | Training-only support and transition triangles, bearing design, 13 fixed policies and compass sectors |
| [inference.py](inference.py) | Joint background/node likelihood, exact and bounded observations, posterior fields and sampler |
| [integration.py](integration.py) | Whole-Earth quadrature, five disjoint support regions × eight bearings, area and refinement checks |
| [estimate.py](estimate.py) | Volume summaries and numerical diagnostics |
| [study.py](study.py) | Source/input/settings signatures, checked caches, fits, retries and geographic/quadrant holdouts |
| [contours.py](contours.py) | Shared complete-contour support audit for this fitted surface |
| [source_cap.py](source_cap.py) | Positive unobserved proximal-cap sensitivity and training-only heldout checks |
| [reporting.py](reporting.py) | Case runner, isopach fragments, figures, report and artifact checksums |

Shared code includes `core/` observations/distributions, `support.py` geometry, regional fold definitions and spherical-mixture sampling/geometry helpers. Those dependencies supply common infrastructure; they do not import the spherical-mixture posterior as evidence. The local interpolation is piecewise linear in **log thickness**, not cubic interpolation.

The default case workflow writes `results/directional/` beside `results/spherical-mixture/`. Standalone: `tephra-isopachs global ERUPTION --geometry directional --output results/new-directional`. Read [output definitions](../../../../docs/OUTPUTS.md) before interpreting posterior arrays or support partitions.

For changes, run [directional numerical tests](../../../../tests/test_directional.py), the matched studies in the [benchmark catalogue](../../../../benchmarks/README.md), and `python -m benchmarks.verify_directional`. Keep every fixed sensitivity, failed attempt, independent replication and known-volume coverage failure. Fixed geometry assumptions and priors are part of the model, not observation-derived facts.
