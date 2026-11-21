# Mass-normalized spherical mixture

The spherical mixture fits kernel amplitudes, locations and shapes to audited thicknesses and bounds. Its principal field lets a fan continue into unsampled directions. The broad parent supplies the kernel posterior and the matched control. [Scientific protocol](../../../../docs/methods/spherical-bearings.md) · [Published maps and checks](../../../../comparisons/spherical-mixture/README.md)

## Follow a fit

1. [Global runner](../../global_reporting.py) resolves the catalogue geometry, fits the baseline and seven prior alternatives, and hands their verified posterior draws to the [bearing report builder](bearings/report.py).
2. [Field](bearings/field.py) defines directional reach from the training sites and the uncertain fan orientation. [Volume](bearings/volume.py) integrates each simulated deposit and separates mass by sector and support class.
3. [Validation](bearings/study.py) uses the training posterior for each held-out region. [Maps](bearings/maps.py) checks contour-area stability and [complete-contour screening](contours.py) replays the fitted surfaces on the shared support protocol.
4. [Report text](bearings/narrative.py) gives the case its principal volume, broad control and warnings. The [cross-eruption builder](../../../../benchmarks/global_posterior.py) combines the case reports with the paired maps and known-volume tests.

## Kernel posterior and geometry

| Module | Purpose |
|---|---|
| [geometry.py](geometry.py) | Equal-area sphere, normalized Gaussian/exponential/power kernels, whole fields and inverse radial CDFs |
| [prior.py](prior.py) | Fixed baseline and seven assumption sensitivities, identical for every eruption |
| [inference.py](inference.py) | Joint likelihood, NumPyro NUTS, multiple-chain diagnostics and reference likelihood |
| [posterior.py](posterior.py) | Exact mixture predictive scores/quantiles, whole-volume summaries and mass partition |
| [study.py](study.py) | Audited input, verified caches, numerical retries and matched geographic holdouts |
| [source_geometry.py](source_geometry.py) | Source-focused directional fans, whole-Earth normalization and angular/radial sampling |
| [source_protocol.py](source_protocol.py) | Named fixed geometry protocols and their sensitivity refits |
| [source_checks.py](source_checks.py) | Source-thickness constraints and posterior normalization refinement |
| [vent_envelope.py](vent_envelope.py) | Experimental bounded field with a vent maximum and smaller secondary peaks |
| [supported_basis.py](supported_basis.py) | Deterministic training-location basis design; no heldout-site leakage |
