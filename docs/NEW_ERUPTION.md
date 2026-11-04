# Add an eruption through chat

[Project home](../README.md) · [Eruption cases](../eruptions/README.md) · [Input schema](../data/README.md)

Upload the measurements and available source material in chat and identify the eruption. A spreadsheet, CSV or published table can be the starting point. You do not need to run a browser app, choose a model, place pseudo-data or set spline smoothing.

## What to supply

- Thickness observations and their locations, with units and coordinate reference if known.
- The source/vent reference and its provenance.
- Paper/table/page references, measurement uncertainties, incomplete deposits, detection limits, and any known redeposition or correlation issues.
- For several fallout fans or phases, identify them and state the intended deposit quantity. Minoan currently combines all accepted Plinian and co-ignimbrite fallout fans in one field.

Use the optional [intake notes](../templates/eruption/INTAKE.md) and [empty CSV schema](../templates/eruption/observations.csv). Unknown information stays explicitly unknown; it is not replaced with invented values. Missing deposits are not automatically zeros or non-detections.

## What the assistant does

1. **Preserve and audit the evidence.** Keep original sources in `data/sources/<slug>/`, record units and coordinate conversions, and document row-level acceptance/exclusion/bound decisions in `data/audits/<name>/`. Record source and accepted-input hashes. Resolve material ambiguities from supplied sources or ask in chat if they cannot be resolved.
2. **Prepare one accepted input.** Write `data/eruptions/<Name>_tephra.csv` using the strict schema. Record the case in `data/datasets.json`; an audited case includes its audit file and case key. Original sources remain unchanged.
3. **Run every active model.** Use `tephra-isopachs compare` into a fresh local `results/` directory. Each model sees the same accepted evidence, outer geographic folds and regional observation hull. Tuning uses training folds only; no preferred method is specified.
4. **Check and interpret.** Inspect maps, withheld predictions, coverage, selection boundaries, volume integration and correlated-draw checks. Keep unfavorable results and failures visible. Report regional bulk fallout; any wider extrapolation needs a separate, explicitly scoped analysis.
5. **Publish a case home on GitHub.** Put reviewed output in `eruptions/<slug>/results/` and create the case `README.md` linking data, audit, results and history. Update the eruption index and, when rerun with a common protocol, the cross-eruption comparison. Run tests and navigation checks before publishing.

If a case is revised, preserve its previous result under `archive/reports/<slug>/<dated-run>/`, retain the original provenance, and explain the change. Avoid overwriting a result while presenting it as the same calculation.

## Local equivalent

```sh
tephra-isopachs compare path/to/observations.csv --output results/new-eruption --name 'Eruption name'
```

The generated report contains all active methods. Reviewed artifacts include `README.md`, figures, `report.json`, accepted/held-out rows and candidate tables. Large regenerable prediction grids remain local. [Output definitions](OUTPUTS.md) · [Installation](QUICKSTART.md).

Automatic fitting does not settle geological source selection or guarantee reliable uncertainty. The report must retain those limits alongside the numerical results.
