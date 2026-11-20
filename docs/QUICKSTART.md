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
```

On Windows, activate with `.venv\Scripts\activate`. Named eruptions are case-insensitive. `python -m src.empirical_isopach.cli` is an equivalent entry point. A nonempty output directory is refused.

The comparison fits the **regional GP and cubic spline peers**, repeats fitting inside geographic holdouts, integrates the common hull, and then runs the **global spherical-mixture posterior**, fixed prior sensitivities and matched holdouts. The directional ensemble allows uncertain fan direction across six structures. Its headline is the whole-Earth integral of the quantitative median map, accompanied by a conditional interval from complete deposits. It retains the single-structure sensitivities and geographic and quadrant tests. Open `directional/README.md` or `spherical-mixture/README.md` directly. [Principal directional protocol](methods/uncertain-azimuth.md). Posterior runs can take substantially longer than interpolation; numerical retries and every assumption refit remain auditable. No per-eruption pseudo-points, smoothing inputs or calculation radius are requested. `compare --regional-only` omits both global companions; the separate `global` command supports `--resume` with strict cache verification.

Before preparing a new input or revising an existing one, review real sites that lack measured thickness. Include source-supported upper or lower limits in the common input and document decisions for the remaining sites. The calculation uses the reviewed CSV and its audit. [Site review before fitting](methods/sites-without-measured-thickness.md)

For the spherical mixture, the catalogue records B-Tm's reviewed `source-fans` geometry; other cases retain the preceding displaced geometry. `global --geometry source-fans` explicitly requests source-focused ellipses for a comparison, while `--geometry displaced` reproduces the preceding geometry. The source maximum is a shape assumption, not a fabricated vent measurement. Read the [audited B-Tm case](../eruptions/b-tm/README.md) before interpreting the competing totals.

## Find results and verify a run

Open the report in the output directory for the regional maps and the links to both global results. The [output guide](OUTPUTS.md) describes the files and volume definitions. For study reproduction or the published checksum and numerical checks, use the [benchmark commands](../benchmarks/README.md).
