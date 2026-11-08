# Tephra fallout reconstruction

**Upload eruption measurements in chat; get audited data, isopach maps, fallout volumes and reproducible reports on GitHub.** Isopachs join places with equal ash thickness.

The workflow is **source evidence → accepted observations → empirical models → validation → published case**. Statistical settings are fitted systematically; no hand-placed pseudo-data or browser application is needed.

| What brings you here? | Start here |
|---|---|
| Read results for an eruption | [Eruption cases](eruptions/README.md) — data, source decisions and three clearly separated reports |
| Understand the project | [Newcomer guide](docs/START_HERE.md) — follow one calculation from evidence to results |
| Supply new measurements | [Upload and publication workflow](docs/NEW_ERUPTION.md) · [Optional intake templates](templates/README.md) |
| Understand the models | [Method catalogue](methods/README.md) — regional GP/spline peers and the global companion |
| Compare performance and uncertainty | [Evidence catalogue](comparisons/README.md) — cross-eruption results, stress tests and targeted studies |
| Work on the repository as an AI agent | [AGENTS.md](AGENTS.md) — sources of truth, task routes, commands and checks |

## Three different questions

| Report | Question | Meaning of its volume |
|---|---|---|
| **Regional reconstruction** | How do GP and cubic spline fit the same observations? | Bulk fallout within their common observation hull; not the complete deposit |
| **Local support audit** | Which locations and contours have nearby observational support? | Partial volume within a diagnosed footprint; gaps and rejected contours stay visible |
| **Conditional global estimate** | What whole-deposit fallout is implied under stated assumptions? | Whole-Earth posterior from a mass-normalized model, with prior sensitivity and numerical checks |

The regional methods are peers, with no automatic winner. The global model uses the same accepted evidence and geographic holdouts but answers a different volume question. A failed contour-support screen does not prevent a conditional global estimate. It highlights why missing deposits and prior assumptions matter. All volumes are bulk fallout, not dense-rock equivalent or total erupted material including ignimbrite deposits.

## Current eruption cases

[Minoan](eruptions/minoan/README.md) · [YTT](eruptions/ytt/README.md) · [Mazama](eruptions/mazama/README.md) · [Tambora](eruptions/tambora/README.md) · [Aso-4](eruptions/aso-4/README.md) · [B-Tm](eruptions/b-tm/README.md)

Each case links its accepted input, source status, current reports and relevant studies. Minoan fits all accepted fallout fans together. Aso-4's source-qualified revision and questioned global total, and B-Tm's source-focused geometry and remaining input qualifications, are explained on their case pages.

## Run the same workflow locally

With Python 3.10 or later, in a checkout:

```sh
python -m pip install '.[test]'
tephra-isopachs methods
tephra-isopachs compare Minoan --output results/minoan-complete
```

`compare` produces regional, support and global reports. Global posterior and sensitivity runs can be lengthy. [Installation and commands](docs/QUICKSTART.md) · [How to read the outputs](docs/OUTPUTS.md).

[Data and provenance](data/README.md) · [Source-code map](src/README.md) · [Repository map](docs/REPOSITORY_MAP.md) · [Contributing](CONTRIBUTING.md) · [Change history](CHANGELOG.md) · [Attribution](THIRD_PARTY_NOTICES.md)
