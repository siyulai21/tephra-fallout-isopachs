# Guides and repository layout

Start at the [eruption catalogue](../eruptions/README.md) for a published result. For a new dataset, use the [intake and publication workflow](NEW_ERUPTION.md).

| Task | Guide |
|---|---|
| Decide whether an occurrence without a measured thickness can enter the fit | [Site decision protocol](methods/sites-without-measured-thickness.md) · [Worked B-Tm evidence](../data/audits/B-Tm/README.md) |
| Install the package and run an analysis | [Quick start](QUICKSTART.md) |
| Interpret a map, volume, interval or output file | [Output guide](OUTPUTS.md) |
| Understand the evidence behind a claim | [Validation guide](VALIDATION.md) |
| Look up equations, priors or diagnostics | [Statistical protocols](methods/README.md) |
| Add a model or contribute a change | [Method contract](ADDING_METHOD.md) · [Contributing](../CONTRIBUTING.md) |
| Track a replaced result or a move | [History](../CHANGELOG.md#earlier-results) · [Maintenance policy](maintenance/README.md) |

## Repository layout

| Location | What belongs there |
|---|---|
| `eruptions/<slug>/` | One case home with all four isopach maps, volumes and predictive scores. Its `results/README.md` contains the matched regional fits. `results/directional/README.md` contains the full directional ensemble report. `results/spherical-mixture/` and `results/support/` contain the independent global posterior and local audit. |
| `data/` | Original sources and hashes, row-level audit decisions and accepted CSVs. [Dataset catalogue](../data/datasets.json) |
| `docs/methods/` | [Four model descriptions, evidence rules and statistical protocols](methods/README.md). |
| `comparisons/` | Cross-eruption validation and simulations. The directional comparison groups the principal fit, its paired control and earlier alternatives. [Directional studies](../comparisons/directional/README.md) |
| `src/` | Shared observation handling, regional registry, model families and reporting. [Code map](../src/README.md) |
| `benchmarks/` and `tests/` | Study reproduction, evidence verifiers and regression tests. [Benchmark commands](../benchmarks/README.md) |
| `templates/` | Optional notes and empty input schema for a new eruption. |
| `results/` at the repository root | Ignored scratch output. Reviewed publications live under their eruption. |
