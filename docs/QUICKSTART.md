# Calculate and compare a new eruption

[Project home](../README.md) · [Input format](../data/README.md) · [Method guide](METHODS.md)

Supply measurements in chat with the eruption name, units, coordinate reference and available source notes. Accepted data, audit decisions and each method report can then be published on GitHub. Missing deposits or illustrative equivalents are not automatically measurements or zeros.

## Run locally

Use Python 3.10 or later, in a repository checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
tephra-isopachs compare Minoan --output results/minoan-complete
tephra-isopachs compare path/to/measurements.csv --output results/new-eruption --name 'New eruption'
tephra-isopachs global Minoan --output results/minoan-global
```

On Windows, activate with `.venv\Scripts\activate`. Named eruptions are case-insensitive. `python -m src.empirical_isopach.cli` is an equivalent entry point. A nonempty output directory is refused.

The comparison fits the **regional GP and cubic spline peers**, repeats fitting inside geographic holdouts, integrates the common hull, and then runs the **global spherical-mixture posterior**, fixed prior sensitivities and matched holdouts. The global reports go in `global/`. Posterior runs can take substantially longer than interpolation; numerical retries and every assumption refit remain auditable. No per-eruption pseudo-points, smoothing inputs or calculation radius are requested. `compare --regional-only` omits the global companion; the separate `global` command supports `--resume` with strict cache verification.

The source data still need defensible units, locations, deposit interpretation and documented bounds. Automatic fitting cannot invent that evidence. All methods retain explicit assumptions and conditional uncertainty limitations.

The catalogue records B-Tm's reviewed `source-fans` geometry; other cases retain the preceding displaced geometry. `global --geometry source-fans` explicitly requests source-focused ellipses for a comparison, while `--geometry displaced` reproduces the preceding geometry. The source maximum is a shape assumption, not a fabricated vent measurement. Read the [B-Tm correction and unresolved Aso-4 comparison](../comparisons/source-revision/README.md) before interpreting the competing totals.

## Reproduce all current evidence

```sh
python -m benchmarks.regional --output results/regional-methods
python -m benchmarks.global_posterior --output results/global-posteriors
python -m benchmarks.mass_mixture --output results/global-stress
python -m benchmarks.global_stress_report --source results/global-stress --output results/global-stress-report
python -m benchmarks.source_fans --output results/source-stress
python -m benchmarks.source_stress_report --source results/source-stress --output results/source-stress-report
python -m pytest -q
python scripts/check_repository_navigation.py
```

The regional runner checks six accepted eruptions and 40 synthetic trials. The global runner fits all six with each case's reviewed geometry, its baseline, seven alternatives and four geographic holdouts. Each of the displaced and source-focused stress suites has twelve baseline trials plus two declared diffuse-tail alternatives. The source revision reran all regional calculations and replaced B-Tm's global calculation; the other five current global reports retain their preceding execution provenance. Unestimable regional trials, poor global coverage and numerical warnings remain visible.

## Read the outputs

- `README.md`: matched maps, regional volumes and interpretation.
- `report.json`: all fits, nested validation, integration checks and provenance.
- `heldout_predictions.csv`: every scored location and fold.
- `accepted_observations.csv`: source-row IDs and preserved bound semantics.
- `selection/`: every evaluated tuning candidate for final and outer fits.
- `predictions.csv`: generated grid; kept locally rather than committed as a large artifact.
- `global/`: whole-Earth posterior estimates, sensitivity reports, baseline full-field samples and all sensitivity volume samples.

See [output definitions](OUTPUTS.md) and the [regional statistical protocol](methods/regional-comparison.md). No map tiles, API keys or browser application are needed.

## Reproduce older analyses

The legacy `tephra-isopachs fit` command remains available for historical selected/stacked GP and global calculations. Its options and assumptions differ; use the [benchmark catalogue](../benchmarks/README.md) and [historical GP output guide](OUTPUTS.md#historical-gp-workflow). Do not attach historical rectangular or global volumes to the new observation-hull comparison.
