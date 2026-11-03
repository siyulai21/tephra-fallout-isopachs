# New eruption workflow

Upload your measurements in chat with the eruption name, units, location reference and available source notes. The assistant will preserve the source, check which records are measured thicknesses, run the calculation and add the accepted data and report to GitHub. Missing deposits, cryptotephra occurrence intervals and illustrative equivalents are not automatically treated as measured thickness or zero deposition.

Reports are Markdown pages with embedded figures. Start with the [Minoan combined-fallout report](../reports/minoan/README.md). No browser upload application or local server is required.

## Reproduce a calculation

Use Python 3.10 or later, from a repository checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
python -m benchmarks.minoan --output results/minoan
```

On Windows, activate with `.venv\Scripts\activate`. The Minoan command verifies the source workbook, deposit arithmetic, selections and checksums, then runs a single combined-fallout reconstruction, the ensemble comparison and four data sensitivities. Read `README.md` and `report.json` in each run directory. The [published comparison](../reports/minoan/README.md) explains the decision to retain the established default.

For other runs:

```sh
tephra-isopachs fit Minoan --output results/minoan-single --nested-cv
tephra-isopachs fit path/to/measurements.csv --output results/new-eruption --nested-cv
```

Bundled names are case-insensitive. `python -m src.probabilistic_isopach` is equivalent. Existing nonempty output directories are refused. The statistical workflow requires no map tiles, API keys, Basemap or Cartopy.

## Settings and recorded assumptions

| Option | Purpose |
|---|---|
| `--method stacked` | Experimental empirical ensemble; `selected` remains the default |
| `--name "My deposit"` | Report title |
| `--levels 0.1 1 2 5 10 20 50` | Positive increasing contour levels in cm |
| `--map-grid 961` | Refine small features; odd values 65–1201 |
| `--max-extent-km 7000` | Map limit; does not change volume bounds |
| `--bounds XMIN YMIN XMAX YMAX` | Volume rectangle in equal-area kilometres |
| `--volume-grid 48 --draws 1500` | Joint-sampling grid and conditional volume draws |
| `--buffer-km 50` | Separation for spatial validation |
| `--grid 128` | Exported prediction-table resolution |
| `--no-publication` | Compact numerical output for benchmarks |

The Minoan and YTT profiles supply documented validation buffers, integration rectangles, map limits and contour levels. Explicit options override them. Profiles match the actual catalogued file path, never a same-named external CSV. Other deposits retain their previous defaults. All resolved settings are recorded.

The report generator writes Markdown, figures, GIS contours, CSV tables and JSON. Atlas and browser/HTML code has been removed. Large prediction arrays remain in the local run directory and can be regenerated from the recorded command and source/input hashes.

See the [input specification](../data/README.md), [output guide](OUTPUTS.md) and [empirical method](EMPIRICAL_METHODS.md). No hand-entered model weights or invented uncertainty values are required. Unsampled sectors still depend on model assumptions.
