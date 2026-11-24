# Calculate and compare a new eruption

[Project home](../README.md) · [Input format](../data/README.md) · [Method guide](methods/README.md)

Supply measurements in chat with the eruption name, units, coordinate reference and available source notes. Accepted data, audit decisions and each method report can then be published on GitHub. Missing deposits or illustrative equivalents are not automatically measurements or zeros.

## Run locally

Use Python 3.10 or later, in a repository checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
tephra-isopachs compare Minoan --output results/minoan-complete
tephra-isopachs compare path/to/measurements.csv --output results/new-eruption --name 'New eruption'
tephra-isopachs global Minoan --output results/minoan-spherical
tephra-isopachs global Minoan --geometry directional --output results/minoan-directional
tephra-isopachs global Minoan --geometry nested-isopachs --output results/minoan-nested
```

On Windows, activate with `.venv\Scripts\activate`. Named eruptions are case-insensitive. `python -m src.empirical_isopach.cli` is an equivalent entry point. A nonempty output directory is refused.

The GP ensemble and cubic spline tune inside geographic training folds and integrate the same observation hull. Each of the three global methods reports its median-thickness map and the volume obtained by integrating that surface, alongside geographic predictions and the integrated Q25–Q75 thickness maps. Use `--geometry nested-whole-area` for the broad nested sensitivity. [Methods and volume definitions](methods/README.md)

Use `compare --regional-only` for the regional pair. The separate `global` command supports `--resume` with verified posterior caches.

Before preparing a new input or revising an existing one, review real sites that lack measured thickness. Include source-supported upper or lower limits in the common input and document decisions for the remaining sites. The calculation uses the reviewed CSV and its audit. [Site review before fitting](methods/sites-without-measured-thickness.md)

The default spherical geometry uses displaced kernels. The catalogue specifies source-focused kernels for B-Tm. Both geometries use the current bearing continuation. The earlier broad control is saved within the same fit at `bearing/parent-report.json`. [B-Tm source audit and fits](../eruptions/b-tm/README.md) · [Spherical protocol](methods/spherical-bearings.md)

## Find results and verify a run

Open the report in the output directory for the regional maps and the links to the three global results. The [output guide](OUTPUTS.md) describes the files and volume definitions. For study reproduction or the published checksum and numerical checks, use the [benchmark commands](../benchmarks/README.md).
