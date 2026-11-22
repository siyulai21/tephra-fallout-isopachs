# Check the offshore station readings

Start with SO-140-25. In the publisher’s Table 1, the 72 mm lower unit and 16 mm upper unit sum to 88 mm. The accepted observation is 8.8 cm. In the corrected Figure 1 on page 2 of the PDF, find station 25 near 119.64° E, 14.76° N. The number on that map identifies the station. The table supplies its thickness. [Publisher’s Table 1](https://link.springer.com/article/10.1007/s00445-003-0306-x/tables/1) · [Corrected plates](https://doi.org/10.1007/s00445-005-0421-y)

## Eleven beds with published bulk thickness

| Station | Bulk (mm) | Unit I (mm) | Unit II (mm) | Input (cm) | Latitude °N | Longitude °E | Cruise event used for position |
|---|---:|---:|---:|---:|---:|---:|---|
| SO-140-25 | 88 | 72 | 16 | 8.8 | 14.76333 | 119.64000 | GIK18452-2 |
| SO-132-41 | 61 | 47 | 14 | 6.1 | 14.64000 | 119.48083 | SO132_41-2 |
| SO-132-37 | 41 | 14 | 27 | 4.1 | 13.90550 | 119.61283 | SO132_37-2 |
| SO-132-29 | 52 | 36 | 16 | 5.2 | 14.34583 | 118.75250 | SO132_29-2 |
| SO-132-28 | 47 | 33 | 14 | 4.7 | 14.32733 | 118.31550 | SO132_28-2 |
| SO-132-13 | 31 | — | — | 3.1 | 14.57867 | 117.39100 | SO132_13-1 |
| SO-132-14 | 32 | — | — | 3.2 | 14.83567 | 116.90833 | SO132_14-1 |
| SO-140-04 | 33 | — | — | 3.3 | 15.18667 | 116.27100 | GIK18431-1 |
| SO-132-07 | 15 | — | — | 1.5 | 15.10550 | 115.38500 | SO132_7-1 |
| SO-140-03 | 7 | — | — | 0.7 | 15.41733 | 115.08350 | GIK18431-2 |
| SO-132-05 | 4 | — | — | 0.4 | 14.16717 | 113.99983 | SO132_5-2 |

Eight positions come from SO132 and three from SO140B. The cruise register locates a deployment at the matching station. Several stations have multiple casts. At SO-140-03 the listed cast is empty. Its coordinate locates the station and the published bulk bed supplies the 7 mm reading. The last column of the transcription retains alternate deployments. [All measured beds and coordinate matches](../../sources/pinatubo-1991/wiesner-2004-table-1.csv) · [Ship events](../../sources/pinatubo-1991/pangaea-coring-events.csv) · [SO132 register](https://www.pangaea.de/expeditions/events/SO132) · [SO140B register](https://www.pangaea.de/expeditions/events/SO140B)

## Hess station groups

The source reports 30 groups. Twelve supply distinct constraints after repeat visits and uncertain primary beds are reviewed. Nine ranges represent replicate thicknesses and their full endpoints are preserved below. The common five-family input uses their lower endpoints as minima. A separate nested-corridor fit uses both endpoints. [All 30 source groups](../../sources/pinatubo-1991/hess-table-1.csv) · [Original paper](https://doi.org/10.1016/S0377-8398%2801%2900025-1) · [Model sensitivity](../../../eruptions/pinatubo-1991-layer-c/results/source-checks/README.md)

| Evidence ID | Ship stations | Original | Common input (cm) |
|---|---|---|---:|
| H01-07 | OR-455-14B, SO-132-27 | 31–33 mm | ≥3.1 |
| H01-10 | SO-114-3 | >35 mm | ≥3.5 |
| H01-13 | SO-114-9, SO-132-50 | >22 mm | ≥2.2 |
| H01-14 | SO-114-10 | Discussion, thinner than 2 cm | ≤2 |
| H01-16 | SO-114-13 | 1–2 mm | ≥0.1 |
| H01-19 | SO-114-27 | 2–3 mm | ≥0.2 |
| H01-22 | SO-132-8 | 8–9 mm | ≥0.8 |
| H01-23 | SO-132-9 | 5–7 mm | ≥0.5 |
| H01-24 | SO-132-10 | 3–4 mm | ≥0.3 |
| H01-25 | SO-132-11 | 6–7 mm | ≥0.6 |
| H01-26 | SO-132-12A | 14–18 mm | ≥1.4 |
| H01-29 | SO-132-16 | 12–14 mm | ≥1.2 |

The 17922 group illustrates a real source discrepancy. Wiesner and Wang report 2.0 cm at their original coring position. Hess and colleagues pool visits under the same station number and report 1.2–1.4 cm about four kilometres away. The fitted series keeps the original core and records the revisit separately. At 17920 the earlier total is 6.0 cm and the later grouped figure is 6.2–6.5 cm. The two older near-source cores include a finer upper deposit whose timing the original authors question. Their published totals bound the climactic component from above. [Original core table and stratigraphy](https://pubs.usgs.gov/pinatubo/wiesner/index.html) · [Repeat decisions](selection.csv)

## Survey coverage and missing bed measurements

Wiesner and Wang describe 65 surface samples, ten recovered ash beds and a sediment trap. The map shows 37 separately resolvable open symbols. Some marks can cover several samples. The trap flux is retained as a cumulative equivalent. The source gives no common thickness detection limit for the open symbols. [Original survey](https://pubs.usgs.gov/pinatubo/wiesner/index.html) · [Numbered open symbols](survey-coverage/wiesner-1996-coverage.png) · [Their coordinates](survey-coverage/wiesner-1996-open-symbols.csv)

The later study describes 89 stations on SO132 and SO140 plus earlier cruises. We checked 65 named station positions from the public event registers against the corrected plate. Its symbols classify deposition or preservation. Only an independently reported bed thickness or bound enters the fit. At SO132-47, for example, Wetzel describes an estimated 1.5 cm bed subsequently mixed into the sediment. That reconstruction and the map symbol both remain in the audit. [65 named map checks](survey-coverage/wiesner-2004-station-probes.csv) · [Preservation observations](../../sources/pinatubo-1991/context/wetzel-2009-sections.csv) · [Wetzel (2009)](https://doi.org/10.1111/j.1365-3091.2009.01066.x)

The later paper cites cruise reports by Wiesner and Kuhnt (1998, University of Hamburg, IBMC Ref. IIA 943) and Wiesner, Stattegger and Kuhnt (1999, Berichte–Reports, Institut für Geowissenschaften, Universität Kiel, 7). The recoverable material did not include an individual thickness ledger for every station in those cruises or the unpublished observations cited by the authors. The inventory distinguishes recoverable measurements, revisits and coverage-only stations. [Original survey and references](https://doi.org/10.1007/s00445-003-0306-x) · [Full evidence inventory](full-evidence.csv)

The three global families also evaluate the fitted field at all 102 georeferenced marine map probes. The archive inventory supplies no visibility threshold and contributes no extra likelihood term. Co-located probes and quantitative cores can therefore be compared without contributing the same bed twice. [Probe inventory and coordinates](archive-probes.json)
