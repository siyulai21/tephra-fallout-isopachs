# Submit an eruption through chat

Upload the thickness measurements and available source material with the eruption name. A spreadsheet, CSV or published table is enough to start. Include coordinates, thickness units, a vent location and any evidence for incomplete deposits, detection limits or redeposition. For multiple fallout fans, say which deposit the result should represent. Minoan combines its accepted Plinian and co-ignimbrite fallout in a single field.

For a documented search with no continuous bed, supply the core description, eruption correlation, inspection method and any visibility or preservation controls. An archive inventory can accompany a new CSV using `--archives path/to/archives.json`. The directional run publishes measured-fan continuations, site probabilities and sensitivity to those controls. A full comparison applies the same archive readings to the independent spherical posterior. A spherical-only run accepts an explicit archive inventory through the same option. [How archive evidence can challenge an isopach](methods/archive-evidence.md)

The [optional intake template](../templates/README.md) gives the observation schema. Leave unknown values explicit. Decisions about sources and deposit interpretation belong in the audit.

## From upload to published case

1. **Audit the sources.** Preserve originals and hashes in `data/sources/`. Record unit conversions, source-row identities, acceptance decisions and documented bounds in `data/audits/`. Review ambiguous evidence before treating it as a measurement.
2. **Register the accepted input.** Save the reviewed CSV in `data/eruptions/` and register its path, audit and applicable spherical geometry in `data/datasets.json`. Carry the same accepted observations into all four families.
3. **Fit the models in a fresh scratch directory.** The GP ensemble and spline tune inside training folds and integrate a common hull. The directional ensemble fits six fan structures with predictive weights learned outside the target eruption. Retain its single-structure sensitivities and proximal-cap experiment. The spherical mixture supplies an independent posterior and fixed prior refits. Both global families predict the same geographic holdouts. [Directional ensemble protocol](methods/directional-ensemble.md)
4. **Inspect the evidence.** Check mapped observation coverage, the source-region maximum, contour rejections, held-out predictions, sampling diagnostics and integration convergence. Preserve unscored folds, failed fits and prior sensitivity. Identify where a global volume relies on extrapolation even when the integral converges. [Output guide](OUTPUTS.md)
5. **Publish one case home.** Put the reviewed output in `eruptions/<slug>/results/`. On the case home, identify the accepted rows and source audit, include the four-panel isopach map and its results table, and link once to each substantive report. Include any material limitation specific to the eruption. Link the case once from the eruption catalogue. Add a cross-eruption study only when the new calculation supports that comparison.

The complete run writes `four-method-isopachs.png` and opens its output README with the comparison. To reproduce that section for the case home, call `comparison_section(results_path, prefix="results/")` from `src.empirical_isopach.visualization.comparison`. Keep case-specific geology and source decisions beside it.

For a revision, record the prior commit, input and model identity and explain what changed. Preserve the scientific evidence needed to reproduce or challenge the result. [Publication and history policy](maintenance/README.md)

## Local equivalent

```sh
tephra-isopachs compare path/to/observations.csv --output results/new-eruption --name 'Eruption name'
```

The full directional report opens at `directional/README.md` inside the output directory. Its ensemble arrays remain under `directional/ensemble/`. The sibling `spherical-mixture/README.md` contains the independent global fit. The regional report is `README.md` and the local audit is `support/README.md`. [Installation and commands](QUICKSTART.md)
