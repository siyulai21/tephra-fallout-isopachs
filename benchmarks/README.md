# Reproduce workflows and studies

Use `python -m benchmarks.minimum_fit_inputs` to check every registered source-to-fit translation. Use `python -m benchmarks.minimum_fit_results` to replay the matched geographic scores, site medians and zero effect of source annotations in the published revised fits. [Fitting and plotting contract](../docs/methods/reported-minimum-fitting.md)

The nested reach study includes three actual survey or archive transfers and an external Calbuco campaign, with a separately published distal contour. The known-surface stress tests have their own results. [Field validation](../comparisons/nested-reach/field-surveys/README.md) · [Reproduction](../docs/methods/nested-reach.md)

For a new case, run the active families with the [intake workflow](../docs/NEW_ERUPTION.md). The commands here replay published evidence or reproduce method studies. Write new fits to a fresh directory under `results/`.

## Current evaluations

The directional study compares proximal, intermediate and whole-deposit contours across all registered cases. It includes actual geographic deletions, bounded sites, known-volume surfaces and omitted surveys. Replay the principal v6 fields and mixture distributions with `python -m benchmarks.verify_directional_geography`; `python -m benchmarks.verify_directional_multiscale` independently replays the historical v5 study. [Current results](../comparisons/directional/geographic-regularity/README.md) · [Equations and reproduction commands](../docs/methods/directional-multiscale.md)

The nested comparison refits the original whole-area geography and the dispersal footprints to matched observations and folds, and retains the finite-corridor control. The study lists commands for the known deposits, survey rotations, field campaigns, compass deletions and release verification. [Nested footprints across all eruptions](../comparisons/nested-isopachs/geography/README.md)

Rebuild a case from its signed full and fold fits with `python -m benchmarks.extent_review --help`. Recreate primary tables from the families published on each current input with `python -m benchmarks.principal_volumes --catalogue --output results/current-indexes`. The normal eruption commands calculate the continuation and its support audit automatically. The paired geographic and known-deposit comparisons use `benchmarks.extent_challenges`, the external surveys use `benchmarks.extent_surveys`, and the normalized-kernel integration check uses `benchmarks.extent_spherical_check`. [Results and numerical checks](../comparisons/global-extent/README.md)

Replay the principal global plates and physical map integrals with `python -m benchmarks.deposit_maps --output results/deposit-maps --workers 3`. The map replay includes both the registered Pinatubo case and its earlier 137-observation development snapshot. Refresh the matched five-family table with `python -m benchmarks.principal_volumes --output results/principal-volumes.csv`. Each registered eruption appears once in that table.

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.azimuth_continuation --help` | Fit and check uncertain fan directions, geographic and quadrant holdouts, median maps and joint volumes | Principal directional ensemble |
| `python -m benchmarks.directional_ensemble --help` | Fit six structures, calibrate predictive weights, render median fields or run known-volume trials | Fixed-direction comparison |
| `python -m benchmarks.directional --output results/directional-study --reports results/directional-cases` | Six directional cases, 13 fixed policies and geographic/quadrant holdouts; full stress/replication sequence in the [study](../comparisons/directional/single-structure/README.md#reproduce) | Single-structure directional sensitivity |
| `python -m benchmarks.directional_comparison --help` | Assemble matched global comparisons from completed cases, stress trials and replications | Both global families |
| `python -m benchmarks.regional --output results/regional-methods` | Matched regional fits for every registered eruption, nested geographic validation and 40 known-truth attempts | GP ensemble and automatic cubic spline |
| `python -m benchmarks.global_posterior --output results/global-posterior` | Fit the registered cases with bearing continuation, seven prior sensitivities and matched geographic holdouts, then build the spherical comparison | Spherical mixture; reviewed geometry per case |
| `python -m benchmarks.source_fans --output results/source-stress` | Twelve source-focused trials with seeds 20/21, including shape violations and hidden mass, plus two diffuse-tail alternatives | Source-focused ellipses |
| `python -m benchmarks.source_stress_report --source results/source-stress --output results/source-stress-report` | Publish every declared source-focused trial, preserving failed attempts and normalization refinements | Reports the source-focused stress cache |
| `python -m benchmarks.global_stress_report --source results/global-stress --output results/global-stress-report` | Publish all twelve completed global stress trials, including coverage failures and numerical warnings | Reports the existing stress-run cache |
| `python -m benchmarks.mass_mixture --output results/global-stress` | Twelve known-truth global-volume stress trials plus the diffuse-tail sensitivity; fresh seeds after documented development revisions | Mass-normalized spherical mixture |
| `python -m benchmarks.support --output results/support-trials` | Twelve known-truth tests of local footprints, complete-contour gates and hidden extrapolation | Regional GP and cubic spline |
| `python -m benchmarks.support_domain --base results/regional-methods --output results/domain-sensitivity` | Retune spline fitting-rectangle padding while keeping the integration footprint fixed | Spline; unchanged GP reference |

To rebuild the published comparisons from the saved fits, run these commands in the repository root. Each table includes the families published on the current registered input. Fixed model-development studies use the [earlier matched editions](../archive/2026-11-24-case-editions/README.md) where needed. The spherical command writes one report containing the broad controls, principal volumes, prior envelopes, paired maps and 24 known-deposit tests. The directional command reads the uncertain-direction estimates and their matched controls from the same fits.

```sh
python -m benchmarks.principal_volumes --catalogue --output results/principal-catalogue
python -m benchmarks.azimuth_summary --output results/directional-summary
python -m benchmarks.regional --summarize-only --published --output results/regional-summary
python -m benchmarks.global_posterior --summarize-only --published --output results/spherical-summary
```

To rerun the spherical simulations from their seeds, follow the [known-deposit commands](../docs/methods/spherical-bearings.md#integration-validation-and-reproducibility). To verify the published fits and their retained controls offline, run `python -m benchmarks.verify_spherical_bearings`. The [dated inventory](../archive/2026-11-20-spherical-continuation/README.md) gives the full historical check.

Replay the contour screens directly from the published spherical fits with `python -m benchmarks.directional_spherical_support --published --output results/spherical-support`. [Case support packets](../comparisons/support/README.md)

## Verify a publication

For the Pinatubo Layer C and total TBJ airfall editions, replay the geographic score of every observation against the buffered splits, the probabilities of the reported ranges and limits, and the archived sampler attempts. Run `python -m benchmarks.global_case_validation --case pinatubo-1991-layer-c` or `--case ilopango-tbj`. The case pages link the resulting site tables.

| Command | Evidence checked |
|---|---|
| `python -m benchmarks.pinatubo_sensitivity --output results/pinatubo-source-tests` | Pinatubo marine holdout, map-reading holdout, two-sided marine ranges and upper-endpoint regional refits. [Checks and results](../archive/2026-11-24-case-editions/pinatubo-layer-c-208/source-checks/README.md) |
| `python -m benchmarks.pinatubo_field_review --output results/pinatubo-new-sites.csv --volume-output results/pinatubo-review-volumes.csv` | Compare the archived fields at 14 newly read sites and the five median-map volumes before and after both source reviews. [Record of the original fits](../archive/2026-11-22-pinatubo-intake/pre-visual-review/README.md) |
| `python -m benchmarks.pinatubo` | Pinatubo’s measured sections, paired sublayers, marine bounds, map coordinates and accepted decisions. [Visual checks](../data/audits/Pinatubo-1991/VISUAL-CHECK.md) |
| `python -m benchmarks.sakurajima_satsuma` | Satsuma source cells, numerical literature readings, original bounds and eight-sector coverage |
| `python -m benchmarks.verify_azimuth_continuation` | Principal uncertain-direction fields, complete-deposit volumes, exact-site coverage and inspected-site sensitivities |
| `python -m benchmarks.verify_btm_cryptotephra` | Current B-Tm profile integrals, upper-limit identities, four fitted families, matched holdouts and archive-response sensitivity volumes |
| `python -m benchmarks.verify_btm_audit` | Archived 68-site B-Tm fits, folds, four source sensitivities and thinning experiments |
| `python -m benchmarks.verify_bearing_continuation` | Archived 68-site measured-fan and inspected-site tests, sector volumes and the independent spherical update |
| `python -m benchmarks.verify_directional` | The 13 directional policies for each registered eruption, plus the original study’s independent replications and known-volume trials |
| `python -m benchmarks.verify_directional_ensemble` | Earlier fixed-direction maps and complete-field volumes, geographic predictions and the published simulation trials |
| `python -m benchmarks.verify_aso_evidence` | Aso-4 source records, matched validation and posterior attempts |
| `python -m benchmarks.verify_visuals` | Published file hashes, current cross-eruption tables, global map integrals and complete-hull regional maps |

The [SHA-256 inventory](published-evidence.sha256) covers published data, machine-readable results, maps and source files used in regression checks. Verify it from the repository root with `shasum -a 256 -c benchmarks/published-evidence.sha256` or use the Python visual verifier. Earlier suites and their exact inputs can be run at the matching [release commit](../CHANGELOG.md#earlier-results). [Reproduce figures](../docs/maintenance/README.md#checking-a-map)

The directional [reproduction sequence](../comparisons/directional/single-structure/README.md#reproduce) also lists `directional_reference` (spherical sector budgets), `directional_replication` (fresh chains) and `directional_spherical_support` (shared contour audit or replay).

## B-Tm core evidence and directional reach

Rebuild the three audited inputs and the digitized sample table with `python -m benchmarks.btm_cryptotephra --output results/btm-counts`. The [current B-Tm study](../eruptions/b-tm/results/cryptotephra/README.md) gives the four-family run, the weaker archive assumptions and the separate regional footprints.

The earlier [measured-fan study](../comparisons/directional/measured-fans/README.md) tested visible-bed transfer scenarios on 68 directly quantitative sites. Its [field report](../archive/2026-11-19-b-tm-thickness-only/results/directional/bearing/field-evidence.md) retains the original fit identities. The [matching release](https://github.com/siyulai21/tephra-fallout-isopachs/tree/d06ba279911f46767fb8c5bb50cfcc9094adbd90) contains that study’s reproduction commands.

## Targeted source and geometry studies

Start with the [Aso-4 study's complete reproduction sequence](../comparisons/aso-4-audit/README.md#reproduce) for source preparation, coordinate revisions, source-conflict endpoints, marine-setting tests and the original-posterior diagnosis. Its [run catalogue](../comparisons/aso-4-audit/runs/README.md) includes every declared attempt and prior. These are development studies, not a request to adopt every tested alternative.

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.source_review --dataset b-tm --output results/source-review` | Three source-constrained candidates, complete geographic holdouts and numerical checks; also accepts `--dataset aso-4` | Development comparisons, not automatic model selection |

`python -m benchmarks.aso_diagnosis` uses the required pre-audit posterior in [the study reference](../comparisons/aso-4-audit/reference/README.md). `python -m benchmarks.verify_aso_evidence` checks that reference, current source hashes, matched validation and all retained posterior attempts without writing tracked files. Use `--output results/verification.json` to save a fresh record.

## Verify the source tables

Run `python -m benchmarks.ilopango` to verify Unit G thicknesses, bounds, the source ledger and angular coverage. Supply `--verify-manuscript PATH` to check the distal core readings and Greenland coordinates against the accepted manuscript. [Ilopango source audit](../data/audits/Ilopango-TBJ/README.md)

Run `python -m benchmarks.minoan` or `python -m benchmarks.ytt` to verify the published inputs against their workbooks. Both commands check the principal selection and four alternatives. Add `--prepare-only` to rebuild the reviewed tables and provenance. Fit either eruption with the ordinary `tephra-isopachs compare` command. [Minoan audit](../data/audits/Minoan/README.md) · [YTT audit](../data/audits/YTT/README.md)

## Bayesian nested isopach volumes

The dispersal-footprint model is the current nested method. [Model and fitting commands](../docs/methods/nested-footprints.md)

Fit the earlier corridor model with `python -m benchmarks.nested_catalogue --output results/nested-cases`. Pass `--whole-area` for the historical control. Replay its maps and exact-site coverage with `python -m benchmarks.nested_principal_maps --output results/corridor-display`. Run `python -m benchmarks.nested_known --output results/nested-known --with-directional` and `python -m benchmarks.nested_spherical_known --output results/nested-spherical-known` for matched known deposits. Replay the published tables with `python -m benchmarks.nested_summary --output results/nested-comparison` and verify the published packets with `python -m benchmarks.verify_nested_reach` and `python -m benchmarks.verify_nested`. [Results](../comparisons/nested-isopachs/README.md)
