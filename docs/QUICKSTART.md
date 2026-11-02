# Quick start

## Install and upload

Use Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
tephra-isopachs serve
```

On Windows, activate with `.venv\Scripts\activate`. The browser workspace accepts a CSV upload or a bundled example. Preview the measurement counts, vent and exclusions, then run the analysis. Each run has a separate directory under `results/workbench/`, and finished reports remain available after restarting the server.

The default workflow needs no Basemap, Cartopy, API keys or network map tiles. The web server is local to your computer. `--no-open` suppresses automatically opening a browser; `--port 8766` changes the port. Stop with Ctrl+C.

## Explore existing reports

Open `docs/atlas/index.html` in a browser. It links all four deposits and their downloadable files. GitHub does not render HTML pages; the PNGs and CSVs can be viewed there directly.

## Rebuild the atlas

```sh
tephra-isopachs atlas --output results/atlas
tephra-isopachs atlas --datasets Mazama B-Tm --output results/two-deposits
```

This fits the selected datasets, runs nested geographic validation, resolves closed contours, checks finite-region integration and builds an offline index. Allow several minutes. `--skip-validation` omits the independent nested check for exploratory runs; model selection still uses spatial validation. Use a fresh output directory for each run.

## Fit one deposit

```sh
tephra-isopachs fit Tambora --output results/tambora --nested-cv
tephra-isopachs fit data/eruptions/Aso-4_tephra.csv --output results/aso-4 --nested-cv
```

Bundled names are case-insensitive. A CSV path also works. `python -m src.probabilistic_isopach` can replace `tephra-isopachs` in every command.

| Option | Purpose |
|---|---|
| `--name "My deposit"` | Report title |
| `--levels 0.1 1 2 5 10 20 50` | Positive increasing contour levels in cm |
| `--map-grid 961` | Refine small features; odd values 65–1201 |
| `--max-extent-km 7000` | Increase map limit; does not change volume bounds |
| `--bounds XMIN YMIN XMAX YMAX` | Explicit volume rectangle in equal-area kilometres |
| `--volume-grid 48 --draws 1500` | Joint-sampling grid and conditional volume draws |
| `--buffer-km 50` | Separation for spatial validation; insufficient folds fail visibly |
| `--grid 128` | Exported prediction-table resolution |
| `--no-publication` | Compact numerical output for benchmarks |

Default map resolution is 481×481. Joint sampling uses 48×48. Expected and median-surface volumes are independently refined at 64, 128, 256 and, if necessary, 512 cells per side. Failed numerical convergence is reported.

See [the input specification](../data/README.md) and [the output guide](OUTPUTS.md). Thickness is in centimetres, coordinates in WGS84 degrees. Invented pseudo rows are ignored. A non-detection needs a documented positive detection limit; a blank region is not a zero.
