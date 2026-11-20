# Spherical-mixture estimates across eruptions

The principal spherical mixture keeps each case’s registered kernel geometry and fits the accepted thicknesses and bounds. In directions without comparable measurements, a complete deposit can continue broadly or follow one unknown fan. The reported volume is the median integral of these complete deposits over the Earth. A credible interval summarizes the baseline posterior. The sensitivity envelope spans the baseline and seven declared changes to geometry, priors or noise. [Bearing protocol and mass normalization](../../docs/methods/spherical-bearings.md) · [Extent, prediction and known-volume tests](bearings/README.md)

| Eruption | Geometry | Baseline median (km³) | Conditional 95% interval (km³) | Prior-sensitivity envelope (km³) | Baseline checks |
|---|---|---:|---:|---:|---|
| [Aso-4](../../eruptions/aso-4/results/spherical-mixture/README.md) | displaced · **questioned** | 1.07e+03 | 403–2.78e+04 | 380–8.51e+07 | pass |
| [B-Tm](../../eruptions/b-tm/results/spherical-mixture/README.md) | source-fans | 47 | 30.2–276 | 25.2–783 | pass |
| [Mazama](../../eruptions/mazama/results/spherical-mixture/README.md) | displaced | 213 | 140–2.61e+03 | 137–9.27e+04 | pass |
| [Minoan](../../eruptions/minoan/results/spherical-mixture/README.md) | displaced | 40.8 | 24.6–1.99e+03 | 23.1–1.91e+04 | pass |
| [Tambora](../../eruptions/tambora/results/spherical-mixture/README.md) | displaced | 162 | 101–920 | 93–3.85e+03 | pass |
| [YTT](../../eruptions/ytt/results/spherical-mixture/README.md) | displaced | 1.51e+03 | 1.05e+03–4.87e+04 | 1.01e+03–1.09e+05 | pass |

6 of 6 baseline fits passed the sampling and integration checks. Numerical warnings occurred in Aso-4 (diffuse tail, one fan, five fans, gaussian errors), B-Tm (wide geography, five fans, gaussian errors), Mazama (wide geography, one fan, gaussian errors), Minoan (wide geography), Tambora (wide geography, five fans). Their results contribute to the envelope. Aso-4's high volume remains qualified by its source audit. At B-Tm, the source-focused fit includes three interpreted upper bounds. [Aso-4 audit](../aso-4-audit/README.md) · [B-Tm occurrence evidence](../../data/audits/B-Tm/README.md)

![Spherical whole-deposit intervals, prior envelopes and withheld-region scores](comparison.png)

## Predictions in withheld regions

Scores use the same observations, training and withheld rows, geographic folds and buffers within each eruption. Lower scores indicate better prediction at the held-out sites. Each case page links the independent directional result and the local contour support audit.

| Eruption | Spherical score ↓ | GP score ↓ | Spline score ↓ | Spherical exact-site 95% coverage | Holdout sampling |
|---|---:|---:|---:|---:|---|
| Aso-4 | 1.632 | 1.514 | 1.662 | 95.2% | pass |
| B-Tm | 1.487 | 1.459 | 4.295 | 93.8% | **provisional** |
| Mazama | 1.561 | 1.272 | 1.971 | 88.9% | **provisional** |
| Minoan | 1.575 | 1.451 | 1.601 | 94.6% | pass |
| Tambora | 1.477 | 1.620 | 1.548 | 100.0% | pass |
| YTT | 1.756 | 1.934 | 2.153 | 84.8% | pass |

[Estimates and diagnostics](comparison.csv) · [Input hashes and matched validation](report.json) · [All four families](../README.md)

## Geometry and validation

The displaced and source-focused geometries have separate known-deposit trials, including failed fits. Sensitivity to the shape of an unmeasured deposit limits interpretation of whole-Earth volumes. [Displaced-geometry trials](stress-tests.md) · [Source-focused trials](../source-revision/stress-tests.md) · [Geometry study](../source-revision/README.md) · [Model development](https://github.com/siyulai21/tephra-fallout-isopachs/tree/2e45ecbc898139d2b93904667b96a8770245c74d/comparisons/spherical-mixture/development) · [Equations and priors](../../docs/methods/probabilistic-global.md)
