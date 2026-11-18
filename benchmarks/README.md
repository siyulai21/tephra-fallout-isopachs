# Reproduce workflows and studies

Use `python -m benchmarks.directional_ensemble --help` to reproduce structural fits, calibration or ensemble reports. `python -m benchmarks.verify_directional_ensemble` replays published median thickness, complete volumes and geographic tests. [Study](../comparisons/directional-ensemble/README.md)

The directional study refits 13 policies and two geographic deletion schemes per eruption. Its known-volume tests include a deposit that has exactly the same observations as another with a hidden fan. [Results and reproduction commands](../comparisons/directional/README.md).

[Project home](../README.md) · [Installation](../docs/QUICKSTART.md) · [Validation guide](../docs/VALIDATION.md) · [Agent task routes](../AGENTS.md)

For one new eruption, use `tephra-isopachs compare` as described in the [intake workflow](../docs/NEW_ERUPTION.md). These modules reproduce broader method evaluations or specific studies. Run from the repository root into a fresh ignored `results/` directory; reviewed outputs live with cases and comparisons.

## Current evaluations

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.directional --output results/directional-study --reports results/directional-cases` | Six directional cases, 13 fixed policies and geographic/quadrant holdouts; full stress/replication sequence in the [study](../comparisons/directional/README.md#reproduce) | Directional continuation |
| `python -m benchmarks.directional_comparison --help` | Assemble matched global comparisons from completed cases, stress trials and replications | Both global families |
| `python -m benchmarks.regional --output results/regional-methods` | Current six-eruption matched regional fits, nested geographic validation and 40 known-truth attempts | GP ensemble and automatic cubic spline |
| `python -m benchmarks.global_posterior --output results/global-posterior` | Six global posteriors using each case's reviewed catalogue geometry, seven sensitivity refits per eruption, matched geographic holdouts and illustrated reports | Spherical mixture; reviewed geometry per case |
| `python -m benchmarks.source_fans --output results/source-stress` | Twelve source-focused trials with seeds 20/21, including shape violations and hidden mass, plus two diffuse-tail alternatives | Source-focused ellipses |
| `python -m benchmarks.source_stress_report --source results/source-stress --output results/source-stress-report` | Publish every declared source-focused trial, preserving failed attempts and normalization refinements | Reports the source-focused stress cache |
| `python -m benchmarks.global_stress_report --source results/global-stress --output results/global-stress-report` | Publish all twelve completed global stress trials, including coverage failures and numerical warnings | Reports the existing stress-run cache |
| `python -m benchmarks.mass_mixture --output results/global-stress` | Twelve known-truth global-volume stress trials plus the diffuse-tail sensitivity; fresh seeds after documented development revisions | Mass-normalized spherical mixture |
| `python -m benchmarks.support --output results/support-trials` | Twelve known-truth tests of local footprints, complete-contour gates and hidden extrapolation | Regional GP and cubic spline |
| `python -m benchmarks.support_domain --base results/regional-methods --output results/domain-sensitivity` | Retune spline fitting-rectangle padding while keeping the integration footprint fixed | Spline; unchanged GP reference |

## Read-only evidence checks

`python -m benchmarks.refresh_figures --output results/new-figures` replays the recorded fits into fresh visual outputs without retuning or resampling. It checks regional parameter and integral parity and uses unchanged global draws. [Rendering and publication policy](../docs/maintenance/visuals.md). `python -m benchmarks.verify_visuals` verifies current figure provenance and preserved scientific hashes.

`python -m benchmarks.verify_directional` verifies all six directional reports, 78 policy fits, mass partitions, six independent replications and 28 known-volume trials. `python -m benchmarks.verify_aso_evidence` verifies the retained source-revision study and posterior checksums. Both run without refitting or changing published evidence.

The directional [reproduction sequence](../comparisons/directional/README.md#reproduce) also lists `directional_reference` (spherical sector budgets), `directional_replication` (fresh chains) and `directional_spherical_support` (shared contour audit or replay). `directional_development` reproduces earlier experiments; its outcomes are retained as development evidence.

## Targeted source and geometry studies

Start with the [Aso-4 study's complete reproduction sequence](../comparisons/aso-4-audit/README.md#reproduce) for source preparation, coordinate revisions, source-conflict endpoints, marine-setting tests and the original-posterior diagnosis. Its [run catalogue](../comparisons/aso-4-audit/runs/README.md) includes every declared attempt and prior. These are development studies, not a request to adopt every tested alternative.

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.source_review --dataset b-tm --output results/source-review` | Three source-constrained candidates, complete geographic holdouts and numerical checks; also accepts `--dataset aso-4` | Development comparisons, not automatic model selection |
| `python -m benchmarks.aso_input_review --output results/aso-input-review` | Treat the source-confirmed eroded Aso-4 row 55 as a 7 cm lower bound in a separate sensitivity, preserving the principal input | Preceding displaced geometry; no model selection or new holdouts |
| `python -m benchmarks.btm_input_review --output results/btm-input-review` | Use Suigetsu's documented upper bound and omit Kushu's unquantified layer from the thickness likelihood, retaining original row identities | Source-focused geometry; targeted sensitivity, not a complete source audit |

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
