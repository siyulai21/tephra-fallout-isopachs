# Reproduce analyses and evaluate methods

The [Aso-4 evidence study](../comparisons/aso-4-audit/README.md#reproduce) lists the complete source preparation, individual-record, marine-setting, source-conflict and displacement-prior commands, including all retained posterior attempts.

[Project home](../README.md) · [Quick start](../docs/QUICKSTART.md) · [Validation catalogue](../docs/validation/README.md)

Run these commands from the repository root after installing `.[test]`. Use a fresh output directory. Generated runs go to ignored `results/`; reviewed publication artifacts live in `eruptions/<slug>/results/` and `comparisons/`.

| Command | What it does | Method family |
|---|---|---|
| `python -m benchmarks.regional --output results/regional-methods` | Current six-eruption matched regional fits, nested geographic validation and 40 known-truth attempts | GP ensemble and automatic cubic spline |
| `python -m benchmarks.global_posterior --output results/global-posterior` | Six global posteriors using each case's reviewed catalogue geometry, seven sensitivity refits per eruption, matched geographic holdouts and illustrated reports | Mass-normalized global models |
| `python -m benchmarks.source_review --dataset b-tm --output results/source-review` | Three source-constrained candidates, complete geographic holdouts and numerical checks; also accepts `--dataset aso-4` | Development comparisons, not automatic model selection |
| `python -m benchmarks.source_fans --output results/source-stress` | Twelve source-focused trials with seeds 20/21, including shape violations and hidden mass, plus two diffuse-tail alternatives | Source-focused ellipses |
| `python -m benchmarks.source_stress_report --source results/source-stress --output results/source-stress-report` | Publish every declared source-focused trial, preserving failed attempts and normalization refinements | Reports the source-focused stress cache |
| `python -m benchmarks.aso_input_review --output results/aso-input-review` | Treat the source-confirmed eroded Aso-4 row 55 as a 7 cm lower bound in a separate sensitivity, preserving the principal input | Preceding displaced geometry; no model selection or new holdouts |
| `python -m benchmarks.btm_input_review --output results/btm-input-review` | Use Suigetsu's documented upper bound and omit Kushu's unquantified layer from the thickness likelihood, retaining original row identities | Source-focused geometry; targeted sensitivity, not a complete source audit |
| `python -m benchmarks.global_stress_report --source results/global-stress --output results/global-stress-report` | Publish all twelve completed global stress trials, including coverage failures and numerical warnings | Reports the existing stress-run cache |
| `python -m benchmarks.mass_mixture --output results/global-stress` | Twelve known-truth global-volume stress trials plus the diffuse-tail sensitivity; fresh seeds after documented development revisions | Mass-normalized spherical mixture |
| `python -m benchmarks.support --output results/support-trials` | Twelve known-truth tests of local footprints, complete-contour gates and hidden extrapolation | Both active families |
| `python -m benchmarks.support_domain --base results/regional-methods --output results/domain-sensitivity` | Retune spline fitting-rectangle padding while keeping the integration footprint fixed | Spline; unchanged GP reference |
| `python -m benchmarks.global_fallout --output results/global-fallout` | Historical six-eruption selected/stacked GP fits, global integrals, geographic deletions and 30 known-truth trials | GP |
| `python -m benchmarks.minoan --output results/minoan` | Verify source arithmetic and rerun combined fallout, ensemble and source sensitivities | GP |
| `python -m benchmarks.ytt --output results/ytt` | Verify the reviewed principal selection and rerun its comparisons/sensitivities | GP |
| `python -m benchmarks.empirical --output results/empirical --repeats 10` | Four-eruption and synthetic selected-versus-stacked development benchmark | GP |
| `python -m benchmarks.run --output results/benchmark --repeats 10` | Initial GP/fixed-default spline comparisons and separate trend-regularized spline selection diagnostics | GP and spline research |
| `python -m benchmarks.summarize_empirical results/empirical/benchmark.json --output results/comparison` | Summarize an existing ensemble benchmark | Reporting only |

`minoan --prepare-only` and `ytt --prepare-only` rebuild audited input files; they are maintenance operations, not necessary merely to read or run the current accepted data. Normal analysis commands verify the inputs before fitting.

## Reproduction versus historical records

Running current code records current source hashes and uses current report/integration behavior. Published older regional results retain the version, inputs and source hashes that produced them. The [report catalogue](../eruptions/README.md) and [path guide](../docs/REPOSITORY_MAP.md) distinguish current results, regional history and superseded input selections.

`ytt_inputs.py` supplies audit reconstruction used by `ytt.py`. Keep scientific changes separate from report organization. No benchmark selects a method because its volume approaches a preferred published number.
