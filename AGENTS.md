# Working on tephra-fallout-isopachs

Users submit measurements in chat. Audit the sources, run all active families and publish a readable case on GitHub. Start with the [intake workflow](docs/NEW_ERUPTION.md) for a new eruption or the [repository layout](docs/README.md#repository-layout) for an existing one.

## Sources of truth

- Use the [archive evidence protocol](docs/methods/archive-evidence.md) for a securely inspected horizon with dispersed ash or no recognizable bed. Preserve the distinction between a qualitative inspection, a calibrated thickness limit and an unsurveyed direction.

- `data/datasets.json` identifies the accepted CSV, audit and reviewed spherical-mixture geometry. Aso-4 uses its reviewed input. B-Tm uses source-focused spherical fans. Preserve original units, row identities, hashes, bounds and reasons for inclusion or exclusion. Never invent evidence, observation errors, pseudo-data or zero deposits.
- `src/empirical_isopach/models/registry.py` lists the regional GP ensemble and cubic spline. Compare them on the same accepted rows, outer geographic folds and observation hull. Tune inside training folds. [Method contract](docs/ADDING_METHOD.md)
- The global families fit the observations independently. Within `results/directional/`, **`ensemble/report.json` is the current directional calculation**. Its `median_surface.volume_km3` integrates the central isopachs and `coherent_fields.volume_km3` summarizes complete-field volumes. Parent-level `report.json` and `single-structure.md` preserve the single-structure reference and broader sensitivities. Use the target-excluded calibration and retain every fit warning. [Directional ensemble protocol](docs/methods/directional-ensemble.md)
- `results/spherical-mixture/report.json` contains the independent global posterior. Run its fixed sensitivities and matched geographic tests. Preserve separate conditional intervals, prior-sensitivity envelopes and numerical status. [Spherical-mixture protocol](docs/methods/probabilistic-global.md)
- Apply the shared support audit to every family. Keep unsupported holes, rejection reasons and assumptions about proximal and distal fallout. Publish conditional global estimates even when no complete band qualifies. [Support protocol](docs/methods/supported-isopachs.md)

## Run and publish

Use a fresh ignored destination. The normal command runs both regional methods, local support, the directional ensemble and its reference sensitivities, and the spherical mixture.

```sh
tephra-isopachs compare ERUPTION_OR_CSV --output results/new-case
```

Publish reviewed outputs under `eruptions/<slug>/results/`. Use **one case home** for the accepted inputs, the four-panel isopach map and matched results table, the support decision and material source or numerical qualifications. Use the shared comparison section and preserve the case-specific interpretation. Link directly to the regional, directional ensemble, spherical-mixture and support reports. Add one row to `eruptions/README.md`. A new cross-eruption link belongs in `comparisons/README.md` when it answers a distinct scientific question.

Preserve earlier result identities, raw evidence, relevant sensitivity studies, failed attempts and hashes when revising a case. Record the preceding immutable revision and retain any locally required historical evidence. Leave unrelated work in progress untouched. [Publication and retention policy](docs/maintenance/README.md)

## Verify

```sh
python -m pytest -q
python scripts/check_repository_navigation.py
python -m benchmarks.verify_visuals
python -m benchmarks.verify_aso_evidence
python -m benchmarks.verify_directional
python -m benchmarks.verify_directional_ensemble
```

For a navigation change, verify scientific artifacts against their pre-change bytes. For a numerical change, run the matched regional benchmark and the relevant global or support studies. New map releases need visual inspection and replay checks. [Study commands](benchmarks/README.md) · [Figure checks](docs/maintenance/visuals.md)
