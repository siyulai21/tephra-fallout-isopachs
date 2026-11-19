# Reproduce workflows and studies

For a new case, run the four families with the [intake workflow](../docs/NEW_ERUPTION.md). The commands here replay published evidence or reproduce method studies. Write new fits to a fresh directory under `results/`.

## Current evaluations

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.azimuth_continuation --help` | Fit and check uncertain fan directions, geographic and quadrant holdouts, median maps and joint volumes | Principal directional ensemble |
| `python -m benchmarks.directional_ensemble --help` | Fit six structures, calibrate predictive weights, render median fields or run known-volume trials | Fixed-direction comparison |
| `python -m benchmarks.directional --output results/directional-study --reports results/directional-cases` | Six directional cases, 13 fixed policies and geographic/quadrant holdouts; full stress/replication sequence in the [study](../comparisons/directional/single-structure/README.md#reproduce) | Single-structure directional sensitivity |
| `python -m benchmarks.directional_comparison --help` | Assemble matched global comparisons from completed cases, stress trials and replications | Both global families |
| `python -m benchmarks.regional --output results/regional-methods` | Current six-eruption matched regional fits, nested geographic validation and 40 known-truth attempts | GP ensemble and automatic cubic spline |
| `python -m benchmarks.global_posterior --output results/global-posterior` | Six global posteriors using each case's reviewed catalogue geometry, seven sensitivity refits per eruption, matched geographic holdouts and illustrated reports | Spherical mixture; reviewed geometry per case |
| `python -m benchmarks.source_fans --output results/source-stress` | Twelve source-focused trials with seeds 20/21, including shape violations and hidden mass, plus two diffuse-tail alternatives | Source-focused ellipses |
| `python -m benchmarks.source_stress_report --source results/source-stress --output results/source-stress-report` | Publish every declared source-focused trial, preserving failed attempts and normalization refinements | Reports the source-focused stress cache |
| `python -m benchmarks.global_stress_report --source results/global-stress --output results/global-stress-report` | Publish all twelve completed global stress trials, including coverage failures and numerical warnings | Reports the existing stress-run cache |
| `python -m benchmarks.mass_mixture --output results/global-stress` | Twelve known-truth global-volume stress trials plus the diffuse-tail sensitivity; fresh seeds after documented development revisions | Mass-normalized spherical mixture |
| `python -m benchmarks.support --output results/support-trials` | Twelve known-truth tests of local footprints, complete-contour gates and hidden extrapolation | Regional GP and cubic spline |
| `python -m benchmarks.support_domain --base results/regional-methods --output results/domain-sensitivity` | Retune spline fitting-rectangle padding while keeping the integration footprint fixed | Spline; unchanged GP reference |

## Verify a publication

| Command | Evidence checked |
|---|---|
| `python -m benchmarks.verify_azimuth_continuation` | Principal uncertain-direction fields, complete-deposit volumes, exact-site coverage and inspected-site sensitivities |
| `python -m benchmarks.verify_btm_audit` | Audited B-Tm rows and bounds, the four fits, geographic folds, source sensitivities and thinning experiments |
| `python -m benchmarks.verify_bearing_continuation` | Measured-fan and inspected-site tests, sector volumes, preserved shard evidence and independent spherical update |
| `python -m benchmarks.verify_directional` | The six current directional fits and their 78 policy runs, plus the original study’s independent replications and known-volume trials |
| `python -m benchmarks.verify_directional_ensemble` | Earlier fixed-direction maps and complete-field volumes, geographic predictions and the published simulation trials |
| `python -m benchmarks.verify_aso_evidence` | Aso-4 source records, matched validation and posterior attempts |
| `python -m benchmarks.verify_visuals` | Published file hashes and the plotted median thicknesses in all six current cases |

The [SHA-256 inventory](published-evidence.sha256) covers published data, machine-readable results, maps and source files used in regression checks. Verify it from the repository root with `shasum -a 256 -c benchmarks/published-evidence.sha256` or use the Python visual verifier. Earlier suites and their exact inputs can be run at the matching [release commit](../docs/HISTORY.md). [Reproduce figures](../docs/maintenance/visuals.md)

The directional [reproduction sequence](../comparisons/directional/single-structure/README.md#reproduce) also lists `directional_reference` (spherical sector budgets), `directional_replication` (fresh chains) and `directional_spherical_support` (shared contour audit or replay). `directional_development` reproduces earlier experiments; its outcomes are retained as development evidence.

## Targeted source and geometry studies

Start with the [Aso-4 study's complete reproduction sequence](../comparisons/aso-4-audit/README.md#reproduce) for source preparation, coordinate revisions, source-conflict endpoints, marine-setting tests and the original-posterior diagnosis. Its [run catalogue](../comparisons/aso-4-audit/runs/README.md) includes every declared attempt and prior. These are development studies, not a request to adopt every tested alternative.

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.source_review --dataset b-tm --output results/source-review` | Three source-constrained candidates, complete geographic holdouts and numerical checks; also accepts `--dataset aso-4` | Development comparisons, not automatic model selection |
| `python -m benchmarks.aso_input_review --output results/aso-input-review` | Treat the source-confirmed eroded Aso-4 row 55 as a 7 cm lower bound in a separate sensitivity, preserving the principal input | Preceding displaced geometry; no model selection or new holdouts |
| `python -m benchmarks.btm_input_review --output results/btm-input-review` | Replay the two-site test in the pinned 84-record B-Tm release | [Historical input and code](https://github.com/siyulai21/tephra-fallout-isopachs/tree/2a52e062b7db84699d5229a25b01282c10301b3e) |

`python -m benchmarks.aso_diagnosis` uses the required pre-audit posterior in [the study reference](../comparisons/aso-4-audit/reference/README.md). `python -m benchmarks.verify_aso_evidence` checks that reference, current source hashes, matched validation and all retained posterior attempts without writing tracked files. Use `--output results/verification.json` to save a fresh record.

## Historical reproduction utilities

These commands are retained for compatibility, audit preparation and regression work; they are not the new-eruption workflow. Old published results are available in [Git history](../docs/HISTORY.md), not a second active report collection.

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.global_fallout --output results/global-fallout` | Historical six-eruption selected/stacked GP fits, global integrals, geographic deletions and 30 known-truth trials | GP |
| `python -m benchmarks.minoan --output results/minoan` | Verify source arithmetic and rerun combined fallout, ensemble and source sensitivities | GP |
| `python -m benchmarks.ytt --output results/ytt` | Verify the reviewed principal selection and rerun its comparisons/sensitivities | GP |
| `python -m benchmarks.empirical --output results/empirical --repeats 10` | Four-eruption and synthetic selected-versus-stacked development benchmark | GP |
| `python -m benchmarks.run --output results/benchmark --repeats 10` | Initial GP/fixed-default spline comparisons and separate trend-regularized spline selection diagnostics | GP and spline research |
| `python -m benchmarks.summarize_empirical results/empirical/benchmark.json --output results/comparison` | Summarize an existing ensemble benchmark | Reporting only |

`minoan --prepare-only` and `ytt --prepare-only` reconstruct audited inputs and are maintenance operations. Normal analysis commands verify existing accepted data. `ytt_inputs.py` supplies the reviewed YTT reconstruction. A report made with current code must record current hashes; do not relabel a new run as the original historical execution.

## Archive evidence and directional reach

The [six-eruption withholding study](../comparisons/directional/measured-fans/README.md) tests all shared range ratios. For a complete case run, use `tephra-isopachs compare B-Tm --output results/btm-archive-replay`. The case catalogue selects its audited archive inventory. To repeat the independently initialized spherical check, use `python -m benchmarks.btm_spherical_archive_refit --work results/btm-spherical-archive-check`. Replay the published median fields, volumes and archive weights with `python -m benchmarks.verify_bearing_continuation`.

After those two fits, regenerate the field interpretation, compare the held-out shard magnitudes and attach the independent spherical refit with `python -m benchmarks.btm_archive_report --case results/btm-archive-replay --spherical-refit results/btm-spherical-archive-check`. The numerical comparisons retain the queried site, source statistic, conversion, original posterior identities and whether the model used the archive to constrain bed visibility. [B-Tm field audit](../eruptions/b-tm/results/directional/bearing/field-evidence.md)
