# Tephra fallout reconstruction

**Turn scattered volcanic-ash thickness measurements into maps of the deposit, estimates of fallout volume, and checks of how much the answer depends on missing data and modeling assumptions.** An *isopach* is a line connecting equal deposit thickness.

The repository contains accepted datasets, source audits, reproducible Python calculations, and illustrated reports for six eruptions. You can read the results on GitHub without installing anything.

## Start here

| I want to… | Go to |
|---|---|
| Understand the project in a few minutes | [Newcomer guide](docs/START_HERE.md) |
| Read an eruption's results | [Report catalogue](reports/README.md) |
| Compare all six eruptions | [Current volume comparison](reports/global-fallout/README.md) |
| Understand GP versus spline methods | [Which method does what?](docs/METHODS.md) |
| Calculate a new eruption or reproduce a run | [Quick start](docs/QUICKSTART.md) |
| Inspect measurements and source decisions | [Data guide](data/README.md) |
| Find a particular part of the code | [Source-code map](src/README.md) |

## What method produces the reports?

**Every current eruption report uses a Gaussian-process (GP) model:** a fitted directional thinning trend plus spatial variation. `selected` chooses one GP configuration using geographic holdouts; `stacked` combines several GP configurations. Neither label means cubic spline.

The [original cubic spline](src/spline_isopach/README.md) and [experimental trend-regularized cubic spline](src/research/README.md) are retained separately for research. They did not produce the current eruption maps and volumes. Their earlier development benchmarks are clearly labelled in the [validation archive](docs/validation/README.md).

## Explore an eruption

| Eruption | Measurements / lower bounds | Report | Input provenance |
|---|---:|---|---|
| Minoan | 92 / 9 | [All fallout fans together](reports/minoan/README.md) | [Source-table audit](data/audits/Minoan/README.md) |
| YTT | 46 / 0 | [Youngest Toba Tuff](reports/ytt/README.md) | [Reviewed principal selection](data/audits/YTT/README.md) |
| Mazama | 45 / 0 | [Reconstruction](reports/mazama/README.md) | [Supplied input flags](data/eruptions/README.md) |
| Tambora | 26 / 0 | [Reconstruction](reports/tambora/README.md) | [Supplied input flags](data/eruptions/README.md) |
| Aso-4 | 87 / 0 | [Reconstruction](reports/aso-4/README.md) | [Supplied input flags](data/eruptions/README.md) |
| B-Tm | 84 / 0 | [Reconstruction](reports/b-tm/README.md) | [Supplied input flags](data/eruptions/README.md) |

Each eruption opens with its current results. `global/` contains current complete-Earth calculations; `regional/` preserves the earlier finite-rectangle results and their sensitivities. YTT also has `archive/` for superseded input selections. **These folders describe calculation scope and history, not different model families.**

![Comparison of modeled fallout volumes and extrapolation across six eruptions](reports/global-fallout/comparison.png)

The headline is **model-conditional bulk fallout integrated over the complete Earth**. A complete integration domain does not mean the deposit is well constrained everywhere. Distant tails, unsampled sectors and preservation can dominate the answer; the reports show these failures. Bulk fallout is not automatically total erupted volume or dense-rock equivalent. [How to read the outputs](docs/OUTPUTS.md).

## Run a calculation

With Python 3.10 or later, from a repository checkout:

```sh
python -m pip install '.[test]'
tephra-isopachs fit Minoan --output results/minoan-example --nested-cv
```

Read `results/minoan-example/README.md`. For your own eruption, supply a CSV in the [documented format](data/README.md), or provide measurements and source notes in chat for review and publication. A new run uses a fresh output directory. The [quick start](docs/QUICKSTART.md) covers installation, custom data and all-eruption reproduction.

## Repository map

| Folder | Contents |
|---|---|
| [data/](data/README.md) | Accepted inputs, unchanged source files, audits and selection history |
| [reports/](reports/README.md) | Published eruption results, current calculations and regional history |
| [docs/](docs/README.md) | Getting started, output interpretation, method derivations and validation |
| [src/](src/README.md) | Active GP workflow, original spline and separate spline research extension |
| [benchmarks/](benchmarks/README.md) | Commands that reproduce analyses and research comparisons |
| [tests/](tests/README.md) | Numerical, statistical and provenance checks |
| `results/` | Local generated runs and large arrays; excluded from Git |

[Contributing](CONTRIBUTING.md) · [Change history](CHANGELOG.md) · [Source and map attribution](THIRD_PARTY_NOTICES.md) · [Folder and path guide](docs/REPOSITORY_MAP.md)
