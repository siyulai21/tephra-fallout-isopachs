# Tephra fallout reconstruction

**From eruption measurements to isopach maps, regional volumes, and reproducible comparisons of empirical statistical models.** An isopach joins locations with the same ash thickness.

The intended workflow is simple: **upload an eruption's data in chat → audit the evidence → run every active method → publish the data and illustrated reports here.** No browser application or manual pseudo-data placement is needed.

| Start with… | What you will find |
|---|---|
| [Eruptions](eruptions/README.md) | One home per eruption: accepted data, source decisions, maps, volumes and reliability checks |
| [Methods](methods/README.md) | The GP ensemble and automatic cubic spline, presented as peers |
| [Comparisons](comparisons/README.md) | Matched regional evaluation across eruptions and known-truth trials |
| [Add an eruption](docs/NEW_ERUPTION.md) | What to upload and how it becomes an auditable GitHub result |
| [Start here](docs/START_HERE.md) | A worked reading path for someone new to the project |

![Regional volumes and withheld prediction scores](comparisons/regional/comparison.png)

## One dataset, multiple methods

Each active model fits the same accepted measurements and documented bounds. Each produces its own thickness surface, isopachs and volume. All use the same geographic holdouts and **regional observation hull**; there is no preferred model or automatic winner.

Today the catalogue contains one **GP ensemble** and one **trend-regularized cubic spline**. The spline selects its knots, tension, smoothing, trend strength and noise automatically. Future empirical families can join the [same comparison contract](docs/ADDING_METHOD.md).

Volumes cover the observation hull, which can contain unsampled gaps and exclude proximal deposits. They are **regional bulk fallout volumes**, not complete eruption volumes or dense-rock equivalents. Conditional uncertainty can understate uncertainty; held-out coverage, geographic-deletion sensitivity and failed checks remain visible.

## Explore the current eruptions

| Eruption | Accepted evidence | Case home |
|---|---|---|
| Minoan | 92 measurements + 9 lower bounds; all accepted fallout fans together | [Data, audit and results](eruptions/minoan/README.md) |
| Youngest Toba Tuff (YTT) | 46 measurements; reviewed source selection | [Data, audit and results](eruptions/ytt/README.md) |
| Mazama | 45 measurements | [Data and results](eruptions/mazama/README.md) |
| Tambora | 26 measurements | [Data and results](eruptions/tambora/README.md) |
| Aso-4 | 87 measurements | [Data and results](eruptions/aso-4/README.md) |
| B-Tm | 84 measurements | [Data and results](eruptions/b-tm/README.md) |

The original four collections retain their supplied flags; they have not received the source-level audits available for YTT and Minoan.

## Reproduce a calculation

With Python 3.10 or later, in a checkout:

```sh
python -m pip install '.[test]'
tephra-isopachs methods
tephra-isopachs compare Minoan --output results/minoan-regional
```

For a new upload, measurements and source notes can be supplied in chat. The assistant prepares accepted input, records source decisions, runs the methods, and adds a case home here. [Upload guide and templates](docs/NEW_ERUPTION.md) · [Local installation](docs/QUICKSTART.md).

## Where things live

| Folder | Role |
|---|---|
| [eruptions/](eruptions/README.md) | Current case homes and reviewed results |
| [data/](data/README.md) | Accepted inputs, unchanged sources and audits |
| [methods/](methods/README.md) | Active method catalogue |
| [comparisons/](comparisons/README.md) | Cross-eruption and known-truth evaluation |
| [docs/](docs/README.md) | User guides, equations and contributor protocol |
| [src/](src/README.md) | Shared workflow and separate statistical model families |
| [archive/](archive/README.md) | Superseded analyses, research benchmarks and reference material |

Local runs go to ignored `results/`. The old `reports/` paths contain navigation pointers for earlier links. Historical calculations retain their original input/code hashes and integration domains.

[Repository map](docs/REPOSITORY_MAP.md) · [Contributing](CONTRIBUTING.md) · [Change history](CHANGELOG.md) · [Attribution](THIRD_PARTY_NOTICES.md)
