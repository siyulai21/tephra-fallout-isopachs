# Repository and path guide

[Project home](../README.md) · [Documentation](README.md)

```text
data/
  eruptions/                  Accepted model inputs
  audits/                     Decisions, sensitivity inputs and selection history
  sources/                    Unchanged published sources
  datasets.json               Named dataset profiles
reports/
  <eruption>/README.md         Start here for an eruption
  <eruption>/comparison/       Current GP-ensemble / cubic-spline regional report
  <eruption>/global/           Historical selected/ensemble GP global reports
  <eruption>/regional/         Earlier finite-region runs and sensitivities
  ytt/archive/                Superseded YTT input selections
  regional-methods/           Current six-eruption and known-truth comparison
  global-fallout/              Earlier GP-only global comparison
  regional-comparison/        Historical four-eruption summary
docs/
  START_HERE.md               Plain-language introduction
  QUICKSTART.md, OUTPUTS.md    How to run and read an analysis
  METHODS.md                  Which model does what
  methods/                    GP, ensemble, volume and spline derivations
  validation/                 Research benchmarks and validation index
  archive/                    Reference images and reporting history
src/
  empirical_isopach/           Active spline and shared regional comparison
  probabilistic_isopach/       Active GP fitter, shared utilities and legacy workflow
  spline_isopach/              Original cubic-spline implementation
  research/                   Historical first trend-penalty experiment
benchmarks/                   Reproduction and evaluation commands
tests/                        Numerical, statistical and data-integrity checks
scripts/                      Repository maintenance checks
results/                      Ignored local runs, not published results
```

## Stable entry points and moved artifacts

The root README and `reports/<eruption>/README.md` remain the entry points. Numerical CSV, JSON, GIS and image artifacts were moved without changing their contents. Paths inside historical provenance record the original execution environment and have deliberately not been rewritten.

| Earlier location | Organized location |
|---|---|
| Regional artifacts at the root of Mazama, Tambora, Aso-4 and B-Tm reports | Each eruption's `regional/` folder |
| Minoan `combined_*` and sensitivity folders | `reports/minoan/regional/` |
| YTT `principal/`, `sensitivity/` and comparison tables | `reports/ytt/regional/` |
| `REGIONAL_REPORT.md` | Short pointer to `regional/README.md` |
| `reports/reference/` | `reports/regional-comparison/` |
| `docs/validation.json` | `docs/validation/initial-benchmark.json` |
| `docs/empirical/` | `docs/validation/ensemble/` |
| Detailed global-volume and ensemble pages | `docs/methods/`; earlier URLs retain pointers |
| Spline extension inside the GP package | `src/research/trend_regularized_spline.py`; old imports remain compatible |

## Naming conventions

`selected` and `stacked` are strategies within the GP family. `global` and `regional` describe the integration domain. `principal` and `sensitivity` describe data selections. The current `spline` label explicitly identifies the automatic cubic-spline method.

Accepted input and source paths stay stable because the catalogue and audits check them. New runs go to a fresh `results/` directory; reviewed outputs are published under `reports/`. See the [contribution guide](../CONTRIBUTING.md).
