# Repository map and sources of truth

[Project home](../README.md) · [Newcomer guide](START_HERE.md) · [Agent task routes](../AGENTS.md)

```text
eruptions/                            Start here for one eruption
  <slug>/README.md                    Accepted evidence and links to every family
  <slug>/results/                     Regional GP/spline comparison and selection
    support/                         Regional local-support audit
    global/README.md                 Reading index for both global families
    directional/                     Directional global posterior and support screen
    spherical-mixture/               Spherical global posterior and sensitivities
data/
  datasets.json                      Accepted input, audit and reviewed spherical geometry
  eruptions/                         Accepted CSVs; required original inputs also retained
  audits/, sources/                  Source decisions, original evidence and hashes
methods/                             One card per active family; relationships in README
comparisons/
  regional/, support/                Matched regional and local-support evidence
  global/README.md                    Shared entry point for both global families
  directional/                       Matched global study, sector tests and replications
    spherical-support/               Retained spherical-family contour-screen evidence
  spherical-mixture/                 Spherical study, stress tests and development record
  aso-4-audit/, source-revision/       Source and geometry studies with all attempts
src/empirical_isopach/
  core/                              Common evidence, probability, geometry and scoring
  models/registry.py                 Regional registry (not a catalogue of global runners)
  models/gaussian_process/            GP ensemble
  models/cubic_spline/                Automatic trend-regularized cubic spline
  models/directional/                 Directional interpolation and global continuation
  models/mass_mixture/                Spherical-mixture posterior and geometry protocols
  cli.py                             Compare every family or run a narrower command
  reporting.py, support_reporting.py  Shared regional/support reporting
  global_reporting.py                Spherical-mixture reporting (historical module name)
benchmarks/                          Reproduction commands, verifiers and targeted studies
tests/                              Numerical/integration regressions and small fixtures
docs/                               User guides, equations and maintenance policy
templates/eruption/                  Optional intake notes and empty schema
scripts/                            Repository navigation checks
results/                            Ignored scratch runs, never the published catalogue
```

The [source map](../src/README.md) separately identifies compatibility packages and their remaining dependencies. There is no browser or atlas workflow.

## Start from the right record

Use the [dataset catalogue](../data/datasets.json), not an arbitrary CSV with a familiar name. It records each accepted input and optional audit. `global_geometry` selects the **spherical-mixture geometry**; the directional family has its own common policies and does not use that setting. Aso-4 uses the reviewed coordinate/bound input. B-Tm's two-site input sensitivity remains separate from its accepted case.

The [method catalogue](../methods/README.md) explains the four families. Their output `report.json` files identify the actual model, input hash, settings, execution and numerical status. Source snapshots and older execution paths are provenance; do not rewrite them to resemble current code.

## Scope and lifecycle

| Material | Home | Policy |
|---|---|---|
| Current reviewed calculation | `eruptions/<slug>/results/` | Revise through a documented run; keep each family's result separate |
| Evidence that supports or challenges a method | `comparisons/` | Retain full protocols, numerical failures and unfavorable outcomes |
| Source evidence and selection history | `data/` | Preserve sources, identities and decisions |
| Required reproduction reference | Beside its study or `tests/fixtures/` | Keep the dependency and checksum explicit |
| Unused old copies or interfaces | [Git history](HISTORY.md) | Remove from the working tree when no current calculation depends on them |

`results/global/` in older provenance means the spherical-mixture calculation now filed under `results/spherical-mixture/`. The standalone command remains `global`; its `--geometry directional` option chooses the other global family. [Commands](QUICKSTART.md) · [Output fields](OUTPUTS.md) · [Cleanup policy](maintenance/README.md).
