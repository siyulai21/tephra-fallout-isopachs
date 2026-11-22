# Pinatubo Layer C intake history

The original intake used 137 rows. Its five fitted models, input, figures and diagnostics remain at their original paths. The expanded case adds marine cores, revises ambiguous land readings and retains documented upper and lower bounds. The comparison before registration included that 137-row case. [Original five fits and input](../../comparisons/nested-isopachs/pinatubo-preview/README.md) · [Earlier comparison table](principal-volumes.csv) · [Current 194-observation case](../../eruptions/pinatubo-1991-layer-c/README.md)

The interim 193-row audit preceded recovery of the SO-114-23 text measurement and its cruise-chart location. The regional and nested fits in this directory reproduce predictions at that independent station from the preceding input. The archived input differs from the final input by precisely that one appended observation. [193-row input](193-site/input.csv) · [Regional fit](193-site/results/report.json) · [Nested field](193-site/results/nested-isopachs/report.json) · [Interim release on GitHub](https://github.com/siyulai21/tephra-fallout-isopachs/tree/9d47ac038b930869dff6628b23b9ccd64681803d/eruptions/pinatubo-1991-layer-c)

Reproduce the held-out comparison from the repository root.

```sh
python -m benchmarks.pinatubo_cruise_check \
  --training-csv archive/2026-11-22-pinatubo-intake/193-site/input.csv \
  --training-results archive/2026-11-22-pinatubo-intake/193-site/results \
  --output results/pinatubo-core-23.json
```

[Published core predictions and the 15 chart-registration checks](../../eruptions/pinatubo-1991-layer-c/results/source-checks/cruise-core-holdout/result.json)
