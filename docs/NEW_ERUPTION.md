# Submit or recalculate an eruption through chat

Upload the thickness measurements and available source material with the eruption name. A spreadsheet, CSV or published table is enough to start. Include coordinates, thickness units, a vent location and any evidence for incomplete deposits, detection limits or redeposition. For multiple fallout fans, say which deposit the result should represent. Minoan combines its accepted Plinian and co-ignimbrite fallout in a single field.

Include the site list even where thickness cells are blank. A site name, core ID, source figure or field description can lead to a usable bound. The person or agent doing the calculation follows those leads and records the decisions. For a recalculation, begin with both the accepted input and the preceding ledger of omissions and proxy values. [How to review sites without measured thickness](methods/sites-without-measured-thickness.md)

Site-derived principal bounds belong in the accepted CSV. A separate archive inventory supplied through `--archives path/to/archives.json` can locate occurrence checks and define visibility sensitivities for the directional and spherical models. A core already fitted through the CSV may appear as a map marker, with its reading counted once. [Archive inventory and modeling options](methods/archive-evidence.md)

The [optional intake template](../templates/README.md) gives the observation schema. Leave unknown values explicit. Decisions about sources and deposit interpretation belong in the audit.

## From upload to published case

1. **Audit the sources.** Preserve original files and hashes and record row identities and units in the case audit. Review each actual site lacking a measured thickness. Trace important locations to their original descriptions, calculate any defensible limits, and document the missing physical quantity or the demonstrated low geometric influence for an omission. Save the calculations and the links from source records to fitted rows. [Site review worksheet](../templates/SITE_EVIDENCE.md)
2. **Register the accepted input.** Save the reviewed CSV in `data/eruptions/` and register its path, audit and applicable spherical geometry in `data/datasets.json`. Distinguish directly measured values, reported bounds and interpreted bounds. Carry the same accepted observations into all four families.
3. **Fit the models in a fresh scratch directory.** The GP ensemble and spline tune inside training folds and integrate a common hull. The directional ensemble allows uncertain fan direction and learns its weights outside the target eruption. Retain the matched fixed-direction control, earlier ensemble calibration and single-structure sensitivities. The spherical mixture supplies an independent posterior and fixed prior refits. Both global families predict the same geographic holdouts. [Directional ensemble protocol](methods/directional-ensemble.md)
4. **Inspect the evidence.** Check mapped observation coverage, the source-region maximum, contour rejections, held-out predictions, sampling diagnostics and integration convergence. Preserve unscored folds, failed fits and prior sensitivity. Identify where a global volume relies on extrapolation even when the integral converges. [Output guide](OUTPUTS.md)
   For influential interpreted sites, compare local thickness, contour reach and volume with the site or linked group omitted and with a weaker response where appropriate. Inspect held-out readings at their own coordinates. Report the principal uncertain-fan-direction median map and its integrated whole-Earth volume first. Keep the fixed-direction and alternate archive responses in the sensitivity section. [Directional extent protocol](methods/uncertain-azimuth.md)
5. **Publish one case home.** Put the reviewed output in `eruptions/<slug>/results/`. On the case home, identify the accepted rows and source audit, include the four-panel isopach map and its results table, and link once to each substantive report. Include any material limitation specific to the eruption. Link the case once from the eruption catalogue. Add a cross-eruption study only when the new calculation supports that comparison.

The complete run writes `four-method-evidence.png` and opens its output README with the comparison. To reproduce that section for the case home, call `comparison_section(results_path, prefix="results/")` from `src.empirical_isopach.visualization.comparison`. Keep case-specific geology and source decisions beside it.

For a revision, record the prior commit, input and model identity and explain what changed. Preserve the scientific evidence needed to reproduce or challenge the result. [Publication and history policy](maintenance/README.md)

## Local equivalent

```sh
tephra-isopachs compare path/to/observations.csv --output results/new-eruption --name 'Eruption name'
```

The full directional report opens at `directional/README.md` inside the output directory. Its principal fields and numeric result are under `directional/azimuth/`. The earlier fixed-direction calibration is in `directional/ensemble/` and the 13-policy reference in `directional/single-structure/`. The sibling `spherical-mixture/README.md` contains the independent global fit. The regional report is `README.md` and the local audit is `support/README.md`. [Installation and commands](QUICKSTART.md)
