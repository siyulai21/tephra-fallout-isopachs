# Mass-normalized spherical mixture

[Scientific protocol and priors](../../../../docs/methods/probabilistic-global.md) · [Results](../../../../comparisons/global/README.md)

| Module | Purpose |
|---|---|
| [geometry.py](geometry.py) | Equal-area sphere, normalized Gaussian/exponential/power kernels, whole fields and inverse radial CDFs |
| [prior.py](prior.py) | Fixed baseline and seven assumption sensitivities, identical for every eruption |
| [inference.py](inference.py) | Joint likelihood, NumPyro NUTS, multiple-chain diagnostics and reference likelihood |
| [posterior.py](posterior.py) | Exact mixture predictive scores/quantiles, whole-volume summaries and mass partition |
| [study.py](study.py) | Audited input, verified caches, numerical retries and matched geographic holdouts |

`global_reporting.py` in the parent framework builds GitHub-readable reports. `tephra-isopachs global` runs the complete protocol; `compare` runs regional peers followed by this companion. No browser, atlas or pseudo-observation interface is involved.
