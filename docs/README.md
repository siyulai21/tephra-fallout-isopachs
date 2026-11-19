# Guides and repository layout

Start at the [eruption catalogue](../eruptions/README.md) for a published result. For a new dataset, use the [intake and publication workflow](NEW_ERUPTION.md).

| Task | Guide |
|---|---|
| Install the package and run an analysis | [Quick start](QUICKSTART.md) |
| Interpret a map, volume, interval or output file | [Output guide](OUTPUTS.md) |
| Understand the evidence behind a claim | [Validation guide](VALIDATION.md) |
| Look up equations, priors or diagnostics | [Statistical protocols](methods/README.md) |
| Add a model or contribute a change | [Method contract](ADDING_METHOD.md) · [Contributing](../CONTRIBUTING.md) |
| Track a replaced result or a move | [History](HISTORY.md) · [Maintenance policy](maintenance/README.md) |

## Repository layout

| Location | What belongs there |
|---|---|
| `eruptions/<slug>/` | One case home with all four isopach maps, volumes and predictive scores. Its `results/README.md` contains the matched regional fits. `results/directional/README.md` contains the full directional ensemble report. `results/spherical-mixture/` and `results/support/` contain the independent global posterior and local audit. |
| `data/` | Original sources and hashes, row-level audit decisions and accepted CSVs. [Dataset catalogue](../data/datasets.json) |
| `methods/` and `docs/methods/` | Introductions to the four empirical models and their detailed protocols. |
| `comparisons/` | Cross-eruption validation, simulations and targeted scientific studies. |
| `src/` | Shared observation handling, regional registry, model families and reporting. [Code map](../src/README.md) |
| `benchmarks/` and `tests/` | Study reproduction, evidence verifiers and regression tests. [Benchmark commands](../benchmarks/README.md) |
| `templates/` | Optional notes and empty input schema for a new eruption. |
| `results/` at the repository root | Ignored scratch output. Reviewed publications live under their eruption. |

[How an inspected core without a measurable bed can test an isopach](methods/archive-evidence.md) · [Six-eruption validation of cross-bearing reach](../comparisons/bearing-continuation/README.md)
