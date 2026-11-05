# Tephra fallout reconstruction

**From eruption measurements to isopach maps, conditional global fallout estimates, and reproducible comparisons of empirical statistical models.** An isopach joins locations with the same ash thickness.

The intended workflow is simple: **upload an eruption's data in chat → audit the evidence → run every active method → publish the data and illustrated reports here.** No browser application or manual pseudo-data placement is needed.

| Start with… | What you will find |
|---|---|
| [Eruptions](eruptions/README.md) | One home per eruption: accepted data, source decisions, maps, volumes and reliability checks |
| [Global fallout estimates](comparisons/global/README.md) | Whole-Earth posterior volumes, explicit assumption sensitivity and unobserved-mass diagnostics |
| [Methods](methods/README.md) | Regional GP/spline peers and the global mass model |
| [Comparisons](comparisons/README.md) | Matched regional evaluation across eruptions and known-truth trials |
| [Which contours can be trusted?](comparisons/support/README.md) | Sampling gaps, contour stability, proximal/distal extrapolation and the limits of global estimates |
| [Add an eruption](docs/NEW_ERUPTION.md) | What to upload and how it becomes an auditable GitHub result |
| [Start here](docs/START_HERE.md) | A worked reading path for someone new to the project |

**Global estimates no longer require a passing contour band.** The old complete-contour screen was too conservative as a prerequisite for conditional extrapolation. A new positive, mass-normalized field estimates total fallout directly on the whole Earth, with no arbitrary calculation rectangle or closing pseudo-data. Its posterior intervals and assumption sensitivities remain explicit: missing deposits cannot be determined from sparse measurements alone. [Results and limitations](comparisons/global/README.md) · [Equations and priors](docs/methods/probabilistic-global.md).

![Global estimates and matched predictive checks](comparisons/global/comparison.png)

## One dataset, multiple methods

Each active model fits the same accepted measurements and documented bounds. Each produces its own thickness surface, isopachs and volume. All use the same geographic holdouts. The two regional methods share integration footprints; the global model integrates over the physical Earth. No automatic winner is selected from the resulting volumes.

For regional reconstruction, the catalogue contains one **GP ensemble** and one **trend-regularized cubic spline**. The spline selects its knots, tension, smoothing, trend strength and noise automatically. The new **mass-normalized spherical mixture** supplies conditional global estimates and is tested on the same geographic holdouts. It fits all accepted fans jointly and varies core thinning, distal tails, fan count, spatial priors, volume priors and observation errors. It is not assumed to outperform both regional families in local prediction.

The support audit removes long sampling bridges and preserves holes, reports three fixed footprint policies, and tests complete contours. It remains a local evidence diagnostic. Its volumes and the broader observation-hull volumes are **partial bulk fallout volumes**; neither is a certified lower bound. The separate global model provides **conditional whole-deposit bulk fallout**, excluding ignimbrite deposits and dense-rock-equivalent conversion. Held-out coverage, known-truth failures and prior dependence remain visible.

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
tephra-isopachs compare Minoan --output results/minoan-complete
# Global companion only (the same fixed priors and sensitivity protocol):
tephra-isopachs global Minoan --output results/minoan-global
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
