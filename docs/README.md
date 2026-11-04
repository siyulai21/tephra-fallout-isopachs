# Documentation

[Project home](../README.md) · [Eruption reports](../eruptions/README.md)

## Understand and use the project

- [Start here](START_HERE.md): the question, workflow, terminology and limitations.
- [New eruption through chat](NEW_ERUPTION.md): upload, audit, calculate and publish.
- [Quick start](QUICKSTART.md): install, run an example, supply new data and reproduce analyses.
- [Read the outputs](OUTPUTS.md): maps, volumes, uncertainty and exported files.
- [Repository map](REPOSITORY_MAP.md): current results, old results and different methods.
- [Data guide](../data/README.md): units, CSV columns and provenance.

## Understand the methods

Start with [the active method catalogue](../methods/README.md) and [method comparison](METHODS.md). Current reports compare the GP ensemble and automatic cubic spline on a shared regional footprint.

- [Gaussian-process model](methods/gaussian-process.md): trend, covariance, censored observations and geographic validation.
- [GP ensemble](methods/gp-ensemble.md): how `stacked` differs from `selected`.
- [Matched regional comparison](methods/regional-comparison.md): common footprint, automatic tuning and nested evidence.
- [Historical global volume integration](methods/global-volume.md): physical domain and extrapolation diagnostics.
- [Cubic-spline methods](methods/cubic-splines.md): automatic knots/smoothing, bounded likelihood and conditional uncertainty.

[Add a model family](ADDING_METHOD.md) explains the common implementation and validation contract.

## Evidence and history

- [Validation catalogue](validation/README.md): current checks and dated development experiments.
- [Historical reference material](../archive/reference/README.md): original map images and retired reporting history.
- [Latest all-eruption analysis](../comparisons/regional/README.md): current results and limitations.

Earlier documentation URLs retain short pointers where a page moved. Numerical records retain the code hashes and paths from the run that produced them.
