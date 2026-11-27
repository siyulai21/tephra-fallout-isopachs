# GP and cubic spline inside the observation hull

Each pair uses the same accepted sites, geographic holdouts and integration area. The volume integrates each model’s median thickness. The table download includes the integrals of the pointwise Q25 and Q75 maps. The table uses the most recent fitted input for each pair. See the [eruption catalogue](../../eruptions/README.md) for coverage across families.

| Eruption | Regional method | Median-map volume (km³) | Held-out score ↓ | Fitted-point 95% coverage |
|---|---|---:|---:|---:|
| [Aso-4](../../eruptions/aso-4/results/minimum-fit-2026-11-25/README.md) | GP ensemble | 170.56 | 1.531 | 93.0% |
| [Aso-4](../../eruptions/aso-4/results/minimum-fit-2026-11-25/README.md) | Trend-regularized cubic spline | 198.18 | 1.694 | 90.7% |
| [B-Tm](../../eruptions/b-tm/results/minimum-fit-2026-11-25/README.md) | GP ensemble | 24.43 | 1.485 | 89.6% |
| [B-Tm](../../eruptions/b-tm/results/minimum-fit-2026-11-25/README.md) | Trend-regularized cubic spline | 51.55 | 4.184 | 82.1% |
| [Ilopango TBJ (total primary airfall)](../../eruptions/ilopango-tbj/results/minimum-fit-2026-11-25/README.md) | GP ensemble | 17.26 | 1.799 | 97.3% |
| [Ilopango TBJ (total primary airfall)](../../eruptions/ilopango-tbj/results/minimum-fit-2026-11-25/README.md) | Trend-regularized cubic spline | 27.62 | 1.757 | 98.7% |
| [Mazama](../../eruptions/mazama/results/minimum-fit-2026-11-25/README.md) | GP ensemble | 60.28 | 1.410 | 94.3% |
| [Mazama](../../eruptions/mazama/results/minimum-fit-2026-11-25/README.md) | Trend-regularized cubic spline | 98.71 | 1.776 | 92.0% |
| [Minoan](../../eruptions/minoan/results/minimum-fit-2026-11-25/README.md) | GP ensemble | 20.38 | 1.542 | 97.0% |
| [Minoan](../../eruptions/minoan/results/minimum-fit-2026-11-25/README.md) | Trend-regularized cubic spline | 24.09 | 1.928 | 92.1% |
| [Pinatubo 1991 (Layer C)](../../eruptions/pinatubo-1991-layer-c/results/minimum-fit-2026-11-25/README.md) | GP ensemble | 4.17 | 1.037 | 98.9% |
| [Pinatubo 1991 (Layer C)](../../eruptions/pinatubo-1991-layer-c/results/minimum-fit-2026-11-25/README.md) | Trend-regularized cubic spline | 5.57 | 0.929 | 95.5% |
| [Sakurajima Satsuma (P14)](../../eruptions/sakurajima-satsuma/results/minimum-fit-2026-11-25/README.md) | GP ensemble | 3.03 | 1.097 | 96.4% |
| [Sakurajima Satsuma (P14)](../../eruptions/sakurajima-satsuma/results/minimum-fit-2026-11-25/README.md) | Trend-regularized cubic spline | 2.97 | 1.103 | 94.5% |
| [Tambora](../../eruptions/tambora/results/README.md) | GP ensemble | 91.14 | 1.620 | 96.2% |
| [Tambora](../../eruptions/tambora/results/README.md) | Trend-regularized cubic spline | 93.46 | 1.548 | 96.2% |
| [YTT](../../eruptions/ytt/results/README.md) | GP ensemble | 1079.21 | 1.934 | 91.3% |
| [YTT](../../eruptions/ytt/results/README.md) | Trend-regularized cubic spline | 943.74 | 2.153 | 82.6% |

Lower negative log predictive scores indicate better predictions in withheld regions. Coverage counts fitted point thicknesses, including source minima fitted as points, within the predictive interval. A dagger marks a numerical volume integration warning. [Regional volumes, thickness bands and held-out scores](comparison.csv)

![Regional median-map volumes and geographic prediction scores](comparison.png)

In the 71-locality B-Tm study, three interpreted upper bounds expand the hull. The spline’s arithmetic mean is dominated by a poorly constrained upper tail near Lake Suigetsu. The case study compares the median map with 20,000 complete-surface draws and reports the integration diagnostics. [Spline sensitivity and regional footprint](../../eruptions/b-tm/results/cryptotephra/README.md#regional-footprint-and-the-splines-upper-tail)

The earlier lower-censored benchmark retains two trials with insufficient training support. [Earlier simulation results](synthetic/README.md) · [Known-deposit benchmark under the current fitting rule](../reported-minima/regional-benchmark/README.md) · [Matched regional protocol](../../docs/methods/regional-comparison.md)

## Reproduce

```sh
python -m benchmarks.regional --summarize-only --published --output results/regional-summary
```
