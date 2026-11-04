# Tephra fallout reconstruction

**Fit isopachs from scattered ash-thickness observations, calculate regional fallout volumes, and compare empirical statistical methods on the same evidence.** An isopach connects locations with equal deposit thickness.

The active workflow reports **one GP ensemble and one trend-regularized cubic spline as peers**, with no preferred method. Both use the same accepted measurements, geographic holdouts and observation footprint. Neither requires invented pseudo-points. Spline smoothing, knots and trend strength are selected automatically.

## Start here

| I want to… | Go to |
|---|---|
| Understand the project | [Newcomer guide](docs/START_HERE.md) |
| Compare the two methods across six eruptions | [Regional comparison and known-truth trials](reports/regional-methods/README.md) |
| Read a particular eruption | [Report catalogue](reports/README.md) |
| Understand GP versus cubic spline | [Method guide](docs/METHODS.md) |
| Calculate a new eruption | [Quick start](docs/QUICKSTART.md) |
| Inspect inputs and source decisions | [Data guide](data/README.md) |
| Find code or historical results | [Source map](src/README.md) · [Repository map](docs/REPOSITORY_MAP.md) |

![Regional volumes and withheld prediction scores](reports/regional-methods/comparison.png)

The volume comparison is restricted to each eruption's **observation hull**, shared by both methods. It can contain unsampled gaps and omit proximal deposits: regional bulk fallout is not total eruption volume or dense-rock equivalent. Distant extrapolation is excluded from method comparison. Conditional intervals, geographic-deletion sensitivity and poor predictive coverage remain explicit.

## Explore an eruption

| Eruption | Exact measurements / lower bounds | Current comparison | Provenance |
|---|---:|---|---|
| Minoan | 92 / 9 | [All fallout fans together](reports/minoan/README.md) | [Source-table audit](data/audits/Minoan/README.md) |
| YTT | 46 / 0 | [Youngest Toba Tuff](reports/ytt/README.md) | [Reviewed selection](data/audits/YTT/README.md) |
| Mazama | 45 / 0 | [Two-method report](reports/mazama/README.md) | [Input flags](data/eruptions/README.md) |
| Tambora | 26 / 0 | [Two-method report](reports/tambora/README.md) | [Input flags](data/eruptions/README.md) |
| Aso-4 | 87 / 0 | [Two-method report](reports/aso-4/README.md) | [Input flags](data/eruptions/README.md) |
| B-Tm | 84 / 0 | [Two-method report](reports/b-tm/README.md) | [Input flags](data/eruptions/README.md) |

## Run a calculation

With Python 3.10 or later:

```sh
python -m pip install '.[test]'
tephra-isopachs compare Minoan --output results/minoan-regional
```

For new data, replace `Minoan` with a CSV path, or supply measurements and source notes in chat for review and publication to GitHub. No browser application or manual statistical tuning is required. Read the generated `README.md` for both methods, shared maps, volumes and nested validation. Existing output directories are protected.

## Repository map

| Folder | Contents |
|---|---|
| [data/](data/README.md) | Accepted inputs, unchanged sources and audits |
| [reports/](reports/README.md) | Current peer comparisons and clearly separated historical results |
| [docs/](docs/README.md) | Guides, statistical definitions and validation |
| [src/](src/README.md) | Regional comparison framework, GP implementation and historical spline code |
| [benchmarks/](benchmarks/README.md) | Reproducible real-data and known-truth comparisons |
| [tests/](tests/README.md) | Numerical, statistical and provenance checks |

Earlier selected-GP, rectangular-region and complete-Earth calculations remain available as historical records. They do not determine the preferred method in this workflow. New empirical families can join the same regional comparison contract.

[Contributing](CONTRIBUTING.md) · [Change history](CHANGELOG.md) · [Attribution](THIRD_PARTY_NOTICES.md)
