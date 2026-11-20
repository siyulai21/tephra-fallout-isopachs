# Spherical-mixture estimates across eruptions

Each case uses its registered source geometry and accepted observations. Every fitted deposit integrates to its own volume over the Earth. A credible interval summarizes the baseline posterior. The sensitivity envelope spans the baseline and seven declared changes to geometry, priors or noise.

| Eruption | Geometry | Baseline median (km³) | Conditional 95% interval (km³) | Prior-sensitivity envelope (km³) | Baseline sampling |
|---|---|---:|---:|---:|---|
| [Aso-4](../../eruptions/aso-4/results/spherical-mixture/README.md) | displaced · **questioned** | 1.47e+03 | 501–4.42e+04 | 465–1.41e+08 | pass |
| [B-Tm](../../eruptions/b-tm/results/spherical-mixture/README.md) | source-fans | 49.1 | 30.7–342 | 25.6–1.23e+03 | pass |
| [Mazama](../../eruptions/mazama/results/spherical-mixture/README.md) | displaced | 223 | 145–3.67e+03 | 142–1.52e+05 | pass |
| [Minoan](../../eruptions/minoan/results/spherical-mixture/README.md) | displaced | 46 | 28.1–2.53e+03 | 26.4–2.12e+04 | pass |
| [Tambora](../../eruptions/tambora/results/spherical-mixture/README.md) | displaced | 179 | 113–1.28e+03 | 104–4.97e+03 | pass |
| [YTT](../../eruptions/ytt/results/spherical-mixture/README.md) | displaced | 1.67e+03 | 1.2e+03–5.49e+04 | 1.14e+03–1.37e+05 | pass |

6 of 6 baseline fits passed the sampling checks. Sampling warnings occurred in B-Tm (five fans, gaussian errors), Mazama (wide geography, gaussian errors). Their results contribute to the envelope. Aso-4's high volume remains qualified by its source audit. At B-Tm, the source-focused fit includes three interpreted upper bounds. [Aso-4 audit](../aso-4-audit/README.md) · [B-Tm occurrence evidence](../../data/audits/B-Tm/README.md)

![Spherical whole-deposit intervals, prior envelopes and withheld-region scores](comparison.png)

## Predictions in withheld regions

Scores use the same observations, training and withheld rows, geographic folds and buffers within each eruption. Lower scores indicate better prediction at the held-out sites. Each case page links the independent directional result and the local contour support audit.

| Eruption | Spherical score ↓ | GP score ↓ | Spline score ↓ | Spherical exact-site 95% coverage | Holdout sampling |
|---|---:|---:|---:|---:|---|
| Aso-4 | 1.632 | 1.514 | 1.662 | 95.2% | pass |
| B-Tm | 1.492 | 1.459 | 4.295 | 93.8% | **provisional** |
| Mazama | 1.577 | 1.272 | 1.971 | 91.1% | **provisional** |
| Minoan | 1.570 | 1.451 | 1.601 | 94.6% | pass |
| Tambora | 1.471 | 1.620 | 1.548 | 96.2% | pass |
| YTT | 1.777 | 1.934 | 2.153 | 84.8% | pass |

[Estimates and diagnostics](comparison.csv) · [Input hashes and matched validation](report.json) · [All four families](../README.md)

## Geometry and validation

The displaced and source-focused geometries have separate known-deposit trials, including failed fits. Sensitivity to the shape of an unmeasured deposit limits interpretation of whole-Earth volumes. [Displaced-geometry trials](stress-tests.md) · [Source-focused trials](../source-revision/stress-tests.md) · [Geometry study](../source-revision/README.md) · [Model development](development/README.md) · [Equations and priors](../../docs/methods/probabilistic-global.md)
