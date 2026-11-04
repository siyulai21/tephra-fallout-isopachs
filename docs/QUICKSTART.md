# Calculate and compare a new eruption

[Project home](../README.md) · [Input format](../data/README.md) · [Method guide](METHODS.md)

Supply measurements in chat with the eruption name, units, coordinate reference and available source notes. Accepted data, audit decisions and both method reports can then be published on GitHub. Missing deposits or illustrative equivalents are not automatically measurements or zeros.

## Run locally

Use Python 3.10 or later, in a repository checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
tephra-isopachs compare Minoan --output results/minoan-regional
tephra-isopachs compare path/to/measurements.csv --output results/new-eruption --name 'New eruption'
```

On Windows, activate with `.venv\Scripts\activate`. Named eruptions are case-insensitive. `python -m src.empirical_isopach.cli` is an equivalent entry point. A nonempty output directory is refused.

The command fits **both** the GP ensemble and cubic spline, repeats fitting inside geographic holdouts, integrates a common observation hull and writes an illustrated `README.md`. It automatically determines spline knots/smoothing, the validation buffer and regional footprint. The only optional public setting is the report title; no per-eruption pseudo-points or smoothing inputs are requested.

The source data still need defensible units, locations, deposit interpretation and documented bounds. Automatic fitting cannot invent that evidence. Both methods retain explicit assumptions and conditional uncertainty limitations.

## Reproduce all current evidence

```sh
python -m benchmarks.regional --output results/regional-methods
python -m pytest -q
python scripts/check_repository_navigation.py
```

This runs six accepted eruptions and ten fixed trials in each of four synthetic scenarios. Unestimable heavily censored trials are retained with failure reasons, not silently dropped. Candidate tables, input/source hashes, folds and numerical checks make the results auditable.

## Read the outputs

- `README.md`: matched maps, regional volumes and interpretation.
- `report.json`: both fits, nested validation, integration checks and provenance.
- `heldout_predictions.csv`: every scored location and fold.
- `accepted_observations.csv`: source-row IDs and preserved bound semantics.
- `selection/`: every evaluated tuning candidate for final and outer fits.
- `predictions.csv`: generated grid; kept locally rather than committed as a large artifact.

See [output definitions](OUTPUTS.md) and the [regional statistical protocol](methods/regional-comparison.md). No map tiles, API keys or browser application are needed.

## Reproduce older analyses

The legacy `tephra-isopachs fit` command remains available for historical selected/stacked GP and global calculations. Its options and assumptions differ; use the [benchmark catalogue](../benchmarks/README.md) and [historical GP output guide](OUTPUTS.md#historical-gp-workflow). Do not attach historical rectangular or global volumes to the new observation-hull comparison.
