# Repository map and sources of truth

[Project home](../README.md) · [Newcomer guide](START_HERE.md) · [Agent task routes](../AGENTS.md)

```text
eruptions/                         Start here for an eruption
  <slug>/README.md                 Evidence, current findings and relevant studies
  <slug>/results/                  Regional GP/spline report and selection tables
  <slug>/results/support/          Local footprints and contour-support diagnostics
  <slug>/results/global/           Conditional global posterior and sensitivities
data/
  datasets.json                   Accepted input and reviewed geometry per case
  eruptions/                      Accepted CSVs; original Aso-4 also retained
  audits/                         Source decisions, qualifications and hashes
  sources/                        Preserved source material
methods/                          One readable card per active model family
comparisons/
  regional/, global/, support/    Cross-eruption results and known-truth evidence
  source-revision/                 B-Tm/Aso-4 geometry and input-qualification studies
  aso-4-audit/                     Source revision, marine tests and uncertainty study
    runs/                         Complete attempts and protocols, including failures
    reference/                    Pre-audit Aso-4 files still used by the diagnosis
src/empirical_isopach/
  core/                           Observations, shared probability and geographic folds
  models/registry.py              Regional GP/spline method registry
  models/gaussian_process/         GP ensemble and its component fitting
  models/cubic_spline/             Automatic trend-regularized cubic spline
  models/mass_mixture/             Global posterior, priors and geometry protocols
  reporting.py, global_reporting.py, support_reporting.py
                                  The three report types
  cli.py                          Public methods/compare/global commands
src/probabilistic_isopach/         Compatibility, legacy GP workflow, reused plot assets
src/spline_isopach/, src/research/  Historical solvers still covered by regression tests
benchmarks/                       Current reproduction and targeted-study commands
tests/fixtures/                   Small reference values needed by regression tests
docs/                            User guides, equations and maintenance policy
templates/eruption/               Optional intake notes and empty measurement schema
scripts/                         Repository navigation checks
results/                         Ignored scratch runs; not published case outputs
```

## Start from the right record

The [dataset catalogue](../data/datasets.json) resolves each accepted input and optional audit, review status or global geometry. The [regional registry](../src/empirical_isopach/models/registry.py) lists GP and spline; the [global protocol code](../src/empirical_isopach/models/mass_mixture/source_protocol.py) defines the separate posterior geometries. Case `report.json` files identify what was actually run.

Aso-4's reviewed input differs from its original study input. B-Tm's principal observations remain unchanged despite a targeted bound sensitivity. These distinctions are recorded in the case and source audit; folder names alone are insufficient to select a dataset.

## Scope and lifecycle

| Material | Home | Policy |
|---|---|---|
| Current reviewed calculation | `eruptions/<slug>/results/` | Update with a documented new run and provenance |
| Evidence needed to justify or challenge a method | `comparisons/` | Retain protocols, numerical failures and unfavorable outcomes |
| Input evidence and selection history | `data/` | Preserve sources and decisions; do not infer new measurements |
| Small required reproduction reference | Beside its study, or `tests/fixtures/` | Keep the dependency explicit and checksum-verifiable |
| Unused old report copies and redirect trees | [Git history](HISTORY.md) | Removed from the current checkout; recover by immutable commit |

There is no current `archive/` or `reports/` collection. [Maintenance policy and cleanup checks](maintenance/README.md).
