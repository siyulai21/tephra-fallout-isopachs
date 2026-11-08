# Repository map

[Project home](../README.md) · [Documentation](README.md)

```text
eruptions/                     START HERE for a particular eruption
  <slug>/README.md             Evidence → results → history
  <slug>/results/              Current reviewed report, maps, tables and provenance
  <slug>/results/global/       Global posterior, assumption refits and sampling checks
  <slug>/results/support/      Local footprint and contour-support diagnostics
data/
  datasets.json               Accepted-case catalogue; stable input paths
  eruptions/                  Accepted measurement CSVs
  audits/                     Row decisions, checksums and selection history
  sources/                    Preserved source files
methods/                       Active model catalogue and plain-language method cards
comparisons/
  regional/                   Current six-eruption and known-truth evaluation
  global/                     Whole-Earth estimates, stress tests and development record
  source-revision/            B-Tm correction, Aso-4 alternatives and source-focused tests
  aso-4-audit/                Aso-4 source corrections, marine checks and geometry diagnosis
  support/                    Shared local-support audit
templates/eruption/            Optional upload notes and empty CSV schema
docs/
  NEW_ERUPTION.md              Chat upload → audit → calculation → GitHub
  ADDING_METHOD.md             Statistical/model integration contract
  methods/                    Equations, assumptions and validation protocol
  organization/               Move manifest and preservation checks
src/empirical_isopach/
  core/                       Observations, probability, folds and survey geometry
  models/registry.py          Regional peers, with no preferred method
  models/gaussian_process/    GP ensemble and its component implementation
  models/cubic_spline/        Automatic trend-regularized cubic spline
  models/mass_mixture/        Mass-normalized spherical posterior and sensitivity studies
  validation.py, region.py    Shared nested comparison and regional volumes
  reporting.py, cli.py        Shared maps, reports and command entry point
  global_reporting.py        Whole-Earth posterior maps, samples and readable reports
src/probabilistic_isopach/    Compatibility imports and historical GP workflow
src/spline_isopach/           Historical original spline/desktop implementation
src/research/                 Historical first trend-penalty experiment
archive/
  reports/                    Superseded eruption calculations and sensitivities
  validation/                 Earlier research benchmarks
  reference/                  Reference images and historical output definitions
benchmarks/                   Reproduction commands; current regional runner first
tests/, scripts/              Numerical, provenance and navigation checks
results/                      Ignored local runs; not a published collection
reports/                      Compatibility navigation pointers only
```

## What changes when a new eruption arrives?

Original source files, audit and accepted CSV are added under `data/`. The accepted case enters `data/datasets.json`. Its reviewed run and readable case home go under `eruptions/<slug>/`; indexes point there. No browser or per-eruption app is created. [Full intake protocol](NEW_ERUPTION.md).

## What changes when a new method arrives?

Its implementation gets a family directory under `src/empirical_isopach/models/`, documentation and validation evidence. Regional families use the registry and shared reporting contract. The global companion has a separate joint-posterior contract, while retaining the same inputs and geographic holdouts. [Full method contract](ADDING_METHOD.md).

## Historical paths

| Earlier location | Current location |
|---|---|
| `reports/<slug>/comparison/` | `eruptions/<slug>/results/` |
| `reports/<slug>/README.md` | `eruptions/<slug>/README.md` |
| `reports/regional-methods/` | `comparisons/regional/` |
| Other numerical `reports/` artifacts | `archive/reports/` |
| `docs/validation/` research artifacts | `archive/validation/` |
| `docs/archive/` reference artifacts | `archive/reference/` |
| GP implementation in `src/probabilistic_isopach/` | `src/empirical_isopach/models/gaussian_process/`; old imports remain wrappers |
| Shared GP data/probability modules | `src/empirical_isopach/core/`; old imports remain wrappers |
| `src/empirical_isopach/spline.py`, `selection.py` | `models/cubic_spline/`; old imports remain wrappers |

Old README URLs retain navigation pointers. The [complete move manifest](organization/path-map.json) records original hashes; numerical files and accepted inputs are preserved. Paths inside historical provenance remain as executed.
