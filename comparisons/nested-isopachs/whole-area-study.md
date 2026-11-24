# Whole-area sensitivity for Bayesian nested isopachs

This study uses the broad whole-area continuation throughout. [Principal corridor estimates and maps](README.md)

Each row compares the same accepted thickness measurements and bounds. All three point estimates integrate median thickness over Earth. Separate conditional intervals use the 2.5th and 97.5th percentiles of the volume draws.

![Median-thickness integrals and conditional intervals](whole-area-volumes.png)

| Eruption | Directional map | Spherical map | Nested map [conditional 95%], km³ | Nested site coverage | Full fit · geographic fits passing |
|---|---:|---:|---:|---:|---|
| [Aso-4](../../eruptions/aso-4/results/nested-whole-area/README.md) | 658.6 | 599.1 | **2,006.6** [648.8–15,969.9] | 75/83 | passed · 3/4 |
| [B-Tm](../../eruptions/b-tm/results/nested-whole-area/README.md) | 33.5 | 37.1 | **67.9** [52.0–534.5] | 60/64 | passed · 4/4 |
| [Ilopango TBJ (Unit G)](../../eruptions/ilopango-tbj/results/nested-whole-area/README.md) | 40.8 | 145.4 | **47.7** [37.5–181.8] | 49/52 | **warning** · 1/4 |
| [Mazama](../../eruptions/mazama/results/nested-whole-area/README.md) | 214.5 | 240.6 | **210.0** [136.0–750.3] | 29/31 | passed · 4/4 |
| [Minoan](../../eruptions/minoan/results/nested-whole-area/README.md) | 28.8 | 29.2 | **31.4** [25.1–91.8] | 85/92 | passed · 4/4 |
| [Pinatubo 1991 (Layer C)](../../eruptions/pinatubo-1991-layer-c/results/nested-whole-area/README.md) | 6.4 | 13.9 | **11.4** [9.8–27.7] | 103/150 | passed · 3/4 |
| [Sakurajima Satsuma (P14)](../../eruptions/sakurajima-satsuma/results/nested-whole-area/README.md) | 7.1 | 7.3 | **11.5** [7.0–60.2] | 49/50 | **warning** · 1/4 |
| [Tambora](../../eruptions/tambora/results/nested-whole-area/README.md) | 95.7 | 126.3 | **123.4** [96.1–290.7] | 26/26 | passed · 4/4 |
| [YTT](../../eruptions/ytt/results/nested-whole-area/README.md) | 900.4 | 1,284.0 | **1,418.8** [1,048.5–3,339.5] | 43/46 | **warning** · 3/4 |

The equal-eruption mean geographic negative log scores are 1.434 for directional, 1.326 for spherical and 1.324 for nested. Each case gives its four buffered regions equal weight. Lower scores indicate better prediction at the withheld sites. Complete contour bands qualified under at least one nested support policy in 0 of 9 cases. Ancient global volumes have no independent measured total here. [Case volumes and scores](whole-area-cases.csv) · [Matched regional rerun](regional-replay.csv)

## Where the nested estimate comes from

| Eruption | Median volume fraction in measured triangles | Median fraction beyond the farthest exact site | That distal fraction among the largest 5% of volumes | 95% interval under exponential / power tails (km³) |
|---|---:|---:|---:|---:|
| Aso-4 | 1.2% | 55.1% | 88.6% | 649–16,417 / 648–13,594 |
| B-Tm | 4.9% | 21.7% | 83.8% | 51–497 / 56–600 |
| Ilopango TBJ (Unit G) | 3.4% | 11.3% | 78.4% | 42–243 / 37–159 |
| Mazama | 19.5% | 3.5% | 2.2% | 134–764 / 142–721 |
| Minoan | 6.7% | 9.5% | 57.0% | 26–95 / 25–87 |
| Pinatubo 1991 (Layer C) | 7.0% | 8.3% | 58.8% | 10–26 / 10–31 |
| Sakurajima Satsuma (P14) | 1.4% | 45.9% | 89.9% | 7–63 / 7–58 |
| Tambora | 8.0% | 5.1% | 15.6% | 98–329 / 96–271 |
| YTT | 13.2% | 0.9% | 4.1% | 1,557–3,445 / 1,012–2,926 |

Each eruption’s sensitivity report retains its prior changes, held-out predictions, areas and conditional 95% volume intervals. [Geographic reach and areas](../nested-reach/README.md) · [Sensitivity intervals for the two global peers](peer-sensitivities.csv)

## Known-deposit validation

| Seeds | Family | Known volumes inside 95% interval | Log interval width | Log interval score | Fits passing |
|---|---|---:|---:|---:|---:|---:|
| 10, 11 | Spherical | 11/12 | 3.835 | 3.870 | 8/12 |
| 10, 11 | Nested | 10/12 | 1.792 | 3.992 | 4/12 |
| 10, 11 | Nested initial | 12/12 | 2.616 | 2.616 | 5/12 |
| 90, 91 | Directional | 9/16 | 0.879 | 2.743 | 5/14 |
| 90, 91 | Spherical | 16/16 | 3.489 | 3.489 | 4/14 |
| 90, 91 | Nested | 14/16 | 1.827 | 2.408 | 3/14 |
| 90, 91 | Nested initial | 16/16 | 2.734 | 2.734 | 3/14 |
| 90, 91 | Directional gap amount | 11/16 | 0.943 | 1.706 | 5/14 |
| 90, 91 | Spherical source prior | 15/16 | 2.693 | 4.635 | 4/14 |

The simulated diffuse distal deposit contained 120 km³. Its conditional 95% interval covered the truth in 0/4 cases. Historical thinning donors provide weak protection against a thin, widespread component. A revised central estimate for Aso-4 or Ilopango cannot establish that this failure mode is absent.

For one identical set of thickness readings, two deposits contain 33.6 and 72.8 km³. The nested fit necessarily gives both the same interval of 30.2–231.1 km³. This tests the ability of an interval to accommodate an unobserved lobe.

Contour recovery is evaluated independently at 1, 10 and 100 cm. The table shows the mean ratio of area correctly predicted above threshold to area above threshold in either the prediction or truth. It includes only paired cases with at least 64 effective quadrature nodes in both unions. The two hidden-lobe truths share the weight of one sampling design. The detailed table records overextended and missed area separately. Pairings use the published geographic quadrature for each suite.

| Seeds | Comparator | Level (cm) | Paired truths | Nested overlap | Comparator overlap |
|---|---|---:|---:|---:|---:|
| 10, 11 | Spherical | 1 | 12 | 0.545 | 0.881 |
| 10, 11 | Spherical | 10 | 12 | 0.905 | 0.924 |
| 10, 11 | Spherical | 100 | 10 | 0.277 | 0.623 |
| 90, 91 | Directional | 1 | 16 | 0.524 | 0.805 |
| 90, 91 | Directional | 10 | 16 | 0.906 | 0.788 |
| 90, 91 | Directional | 100 | 14 | 0.397 | 0.109 |

These designed stress tests reuse published sampling geometries. They exercise failure modes and give no population-level frequency for those modes in nature. The directional, spherical and nested estimates remain separate. A wide conditional interval can cover a true volume even when its median misses a large lobe. The geographically withheld tests and the known-volume tests answer different questions.

[Known-volume predictions and truths](known-volumes.csv) · [False and missed isopach areas](known-areas.csv) · [Raw challenge diagnostics](challenge-receipts.json.gz) · [Nested model and prior calibration](../../docs/methods/nested-isopachs.md)

## Recovery on the Aso-4 and Ilopango survey layouts

These four tests were specified after freezing the final model. Only the actual site coordinates and source locations are used. Every thickness and detection limit is simulated. Each design has the same three-component deposit in units of its median survey distance. The paired version adds a localized positive signal to the observed archive while the true primary fallout volume stays fixed. This tests how easily the method mistakes local thickening for primary ash elsewhere.

| Survey | Local archive thickening | True primary (km³) | Initial 95% interval | Refined 95% interval | Initial / refined checks |
|---|---|---:|---:|---:|---|
| aso-4 | yes | 298.0 | 258.3–661.9 | 260.9–586.9 | passed / warning |
| aso-4 | no | 298.0 | 216.8–627.5 | 210.7–407.3 | warning / passed |
| ilopango-tbj | yes | 1.2 | 0.8–1.5 | 0.8–1.6 | passed / warning |
| ilopango-tbj | no | 1.2 | 0.8–1.5 | 0.7–2.1 | passed / passed |

The synthetic archive signal is a diagnostic scenario. It supplies no evidence that any particular site was redistributed. The earlier panels use all matched trials, including poorly covered and numerically uncertain outcomes. [Simulation field, observation process and reference integrals](../../benchmarks/nested_deposit_recovery.py) · [Regional known-deposit replay](regional-known.csv)
