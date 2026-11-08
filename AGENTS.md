# Agent entry point

## Intended use

The user uploads eruption observations and source material in chat. Audit that evidence, run the current empirical workflow, and publish accepted data and readable reports to GitHub. Keep every method's result and uncertainty separate. There is no browser/atlas workflow and no preferred volume or model to reproduce.

## Read only what your task needs

| Task | Read first | Implementation / output |
|---|---|---|
| Understand the current project | [Newcomer guide](docs/START_HERE.md), [method catalogue](methods/README.md) | [Repository map](docs/REPOSITORY_MAP.md) |
| Analyze or revise an eruption | [Intake protocol](docs/NEW_ERUPTION.md), its [case home](eruptions/README.md), linked source audit | [Dataset catalogue](data/datasets.json); `eruptions/<slug>/results/` |
| Change a regional method | [Regional protocol](docs/methods/regional-comparison.md), [method contract](docs/ADDING_METHOD.md) | [Registry](src/empirical_isopach/models/registry.py), `models/gaussian_process/` or `models/cubic_spline/` |
| Change a global method | [Global protocol](docs/methods/probabilistic-global.md), [source-focused protocol](docs/methods/source-anchored-fans.md) | [Mass model](src/empirical_isopach/models/mass_mixture/README.md), [global reporting](src/empirical_isopach/global_reporting.py) |
| Investigate support or contour closure | [Support protocol](docs/methods/supported-isopachs.md) | [Support calculation](src/empirical_isopach/support.py), [support reporting](src/empirical_isopach/support_reporting.py) |
| Reproduce Aso-4 source/geometry work | [Source audit](data/audits/Aso-4/README.md), [study and commands](comparisons/aso-4-audit/README.md) | [Study runs](comparisons/aso-4-audit/runs/README.md), [required reference](comparisons/aso-4-audit/reference/README.md) |
| Investigate B-Tm/source geometry | [Study](comparisons/source-revision/README.md), [B-Tm case](eruptions/b-tm/README.md) | [Benchmark command catalogue](benchmarks/README.md) |
| Reorganize or edit documentation | [Maintenance policy](docs/maintenance/README.md) | [Navigation checker](scripts/check_repository_navigation.py); keep scientific artifacts unchanged |

## Sources of truth

- `data/datasets.json` chooses each accepted CSV, audit/case key and reviewed global geometry. Aso-4 uses `Aso-4_reviewed.csv`; B-Tm uses `source-fans`. Original/sensitivity inputs are not automatic replacements.
- `src/empirical_isopach/models/registry.py` lists **regional** peers: GP ensemble and automatic cubic spline. The global model has a separate posterior contract; it is not missing from this registry by accident.
- Case `report.json`, global `report.json`, input audits and posterior signatures identify the actual calculation. Read the scope, input hash, geometry and numerical status before using a number. Do not infer current settings from an older study or filename.
- Read only the relevant report/summary first. Large posterior `.npz`, draw tables and complete attempt records are evidence to inspect when needed, not introductory context. Full development records include failures and must not be pruned as obsolete clutter.

## Execution and publication

Use a fresh ignored `results/` directory. The usual command is:

```sh
tephra-isopachs compare ERUPTION_OR_CSV --output results/new-case
```

It runs the regional peers, local support audit and conditional global companion. `compare --regional-only` is an explicitly narrower run. `global ... --resume` resumes only matching checksum-verified caches. Geometry overrides are declared experiments; do not silently override the catalogue or adopt a smaller total.

Publish reviewed output under `eruptions/<slug>/results/`, update its case home and the relevant indexes, and record the source/input/method identity. Aso-4 targeted studies and source-geometry comparisons are development evidence, not independent validation or universal prior selection. Follow [the publication checklist](docs/NEW_ERUPTION.md).

## Scientific constraints

Preserve original sources, hashes, row identities, units and acceptance/exclusion decisions. Bounds are likelihood events; missing or reworked deposits are not zeros. Never invent measurement errors, geological evidence or training pseudo-observations. Fit all accepted fallout fans jointly for the stated deposit quantity.

Tune response-dependent regional settings within training folds and compare on identical accepted evidence, outer geographic folds and regional footprints. Keep observation noise out of latent volumes. Preserve whole-field correlation in uncertainty draws. The global posterior uses whole-Earth mass normalization, declared priors, sensitivity refits and matched holdouts. A support failure is a diagnostic, not a veto on the user-requested conditional global calculation. Numerical convergence is not evidence that missing global mass is identified.

Retain poor coverage, rejected contours, unsupported holes, search boundaries, unsuccessful numerical attempts and proximal/distal assumptions. Conditional intervals, geographic-deletion ranges and prior-sensitivity envelopes are different quantities.

## Verification and history

```sh
python -m pytest -q
python scripts/check_repository_navigation.py
python -m benchmarks.verify_aso_evidence
```

The last command checks published Aso-4 evidence without overwriting it; `--output` can save a new verification record. Match additional checks to the task: numerical changes need the relevant regional/global/support benchmarks; organization changes need hash preservation and path checks. Do not rerun expensive posterior studies just to edit navigation.

Unused old report copies and redirect collections have been removed from the working tree. [Git history](docs/HISTORY.md) preserves them. Keep local reference files only when current reproduction or tests need them. Do not rewrite historical execution hashes to match a new folder layout. Leave unrelated working copies and in-progress work untouched.
