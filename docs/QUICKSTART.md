# Run a Gaussian-process eruption analysis

[Project home](../README.md) · [Newcomer guide](START_HERE.md) · [Method comparison](METHODS.md)

**These commands run the Gaussian-process workflow.** `selected` means one GP candidate; `stacked` means a GP ensemble. The spline research implementations are separate.

Upload your measurements in chat with the eruption name, units, location reference and available source notes. The assistant will preserve the source, check which records are measured thicknesses, run the calculation and add the accepted data and report to GitHub. Missing deposits, cryptotephra occurrence intervals and illustrative equivalents are not automatically treated as measured thickness or zero deposition.

Reports are Markdown pages with embedded figures. Start with the [Minoan combined-fallout report](../reports/minoan/README.md). You can read published reports directly on GitHub without installing the calculation tools.

## Reproduce a calculation

Use Python 3.10 or later, from a repository checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
python -m benchmarks.minoan --output results/minoan
```

On Windows, activate with `.venv\Scripts\activate`. The Minoan command verifies the source workbook, deposit arithmetic, selections and checksums, then runs a single combined-fallout reconstruction, the ensemble comparison and four data sensitivities. Read `README.md` and `report.json` in each run directory. The [published comparison](../reports/minoan/regional/README.md) explains the single-GP/ensemble decision; the [current overview](../reports/minoan/README.md) gives the global headline.

For other runs:

```sh
tephra-isopachs fit Minoan --output results/minoan-single --nested-cv
tephra-isopachs fit path/to/measurements.csv --output results/new-eruption --nested-cv
```

Bundled names are case-insensitive. `python -m src.probabilistic_isopach` is equivalent. Existing nonempty output directories are refused. The statistical workflow requires no map tiles, API keys, Basemap or Cartopy.

## Settings and recorded assumptions

| Option | Purpose |
|---|---|
| `--method stacked` | Experimental GP ensemble; the single-GP `selected` strategy remains the default |
| `--name "My deposit"` | Report title |
| `--levels 0.1 1 2 5 10 20 50` | Positive increasing contour levels in cm |
| `--map-grid 961` | Refine small features; odd values 65–1201 |
| `--max-extent-km 7000` | Map limit; does not change the global integral |
| `--bounds XMIN YMIN XMAX YMAX` | Regional diagnostic rectangle in equal-area kilometres; does not change the global integral |
| `--volume-grid 48 --draws 1500` | Regional joint-sampling grid and conditional draws; not a global interval |
| `--buffer-km 50` | Separation for spatial validation |
| `--grid 128` | Exported prediction-table resolution |
| `--no-publication` | Compact numerical output for benchmarks |

The Minoan and YTT profiles supply documented validation buffers, regional diagnostic rectangles, map limits and contour levels. Explicit options override them. Profiles match the actual catalogued file path, never a same-named external CSV. Other deposits retain their previous defaults. All resolved settings are recorded.

The report generator writes Markdown, figures, GIS contours, CSV tables and JSON. Each report names its GP method. Large prediction arrays remain in the local run directory and can be regenerated from the recorded command and source/input hashes.

See the [input specification](../data/README.md), [output guide](OUTPUTS.md) and [empirical method](methods/gp-ensemble.md). No hand-entered model weights or invented uncertainty values are required. Unsampled sectors still depend on model assumptions.

Run both GP strategies for every catalogued eruption, geographic-deletion checks and 30 synthetic trials:

```sh
python -m benchmarks.global_fallout --output results/global-fallout
```

The [reproduction command catalogue](../benchmarks/README.md) separates current analyses from historical research benchmarks. Source files and inputs stay unchanged when you run an ordinary calculation.
