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
3. **Run regional peers and the global companion.** Use `tephra-isopachs compare` into a fresh local `results/` directory. GP and spline share the accepted evidence, geographic folds and observation hull. The mass-normalized global model follows in `global/`, with fixed generic priors, full sensitivity refits and identical geographic holdouts. Tuning uses training folds only; no published target volume is supplied.
4. **Check and interpret.** Inspect maps, withheld predictions, coverage, selection boundaries, integration and posterior sampling diagnostics. Explicitly inspect the thickest contour and source-region maximum: a finite whole-Earth integral does not prevent an unsupported displaced peak. Compare the [source-focused alternative](methods/source-anchored-fans.md) when this problem appears; its maximum-at-vent assumption is not a universal volcanological law. Keep unfavorable results and failures visible. A lower volume alone never justifies replacement. Report partial regional volumes separately from conditional global bulk fallout. A failed contour-support screen is not a veto on the global posterior; unobserved deposits instead require explicit assumptions and sensitivity reporting.
5. **Publish a case home on GitHub.** Put reviewed output in `eruptions/<slug>/results/` and create the case `README.md` linking data, audit, results and history. Update the eruption index and, when rerun with a common protocol, the cross-eruption comparison. Run tests and navigation checks before publishing.

If a case is revised, identify the preceding immutable Git commit, retain its provenance and explain what changed. Keep a local reference beside the study only when a current calculation or check requires it. Do not duplicate unused archived report trees or present a revised result as the same calculation. [Publication and history policy](maintenance/README.md).

## Local equivalent

```sh
tephra-isopachs compare path/to/observations.csv --output results/new-eruption --name 'Eruption name'
```

The generated report contains the regional peers and a `global/` companion. Reviewed artifacts include readable reports, figures, numerical provenance, heldout predictions, candidate tables, baseline global posterior samples and sensitivity volume samples. Large regenerable prediction grids and auxiliary posterior caches remain local. `tephra-isopachs global` can run the global analysis separately; `--resume` verifies the input/model/settings/checksums before resuming a lengthy calculation. [Output definitions](OUTPUTS.md) · [Installation](QUICKSTART.md).

Automatic fitting does not settle geological source selection or guarantee reliable uncertainty. The report must retain those limits alongside the numerical results.
