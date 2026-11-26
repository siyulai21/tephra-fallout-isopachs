# Spherical bearing validation

The geographic score averaged equally across eruptions changed from 1.485 to 1.477. Of the withheld readings, 274 had gap exposure of at least 0.5. Within those readings the site-weighted score was 1.615 → 1.593. Giving each represented eruption equal weight gives 1.796 → 1.814. [Paired predictions and gap-exposure scores](paired-cases.csv)

## Where the map changes

![Broad and principal spherical 10 cm isopachs by eruption](paired-isopachs.png)

Each panel compares the same 10 cm threshold. Tan shading identifies ground where only the earlier median reaches 10 cm. Blue shading identifies ground where both medians reach it. Only sites inside each frame are visible. The area calculations below cover the entire Earth.

| Eruption and compass sector | Area ≥10 cm, earlier → principal (km²) | Split-draw area disagreement |
|---|---:|---:|
| Aso-4, S | 361,088 → 217,253 | 4.1% |
| Aso-4, SE | 349,340 → 257,326 | 4.4% |
| Aso-4, W | 51,952 → 24,082 | 2.5% |
| Mazama, SW | 1,497 → 1,047 | 0.0% |

These are modelled areas in 45° sectors, averaged over two scrambled whole-Earth integration grids. The sites do not establish the actual deposit edge. The downloadable tables give both grid results, contour-area split checks and all eight bearings at 1, 10 and 100 cm.

## Deposits with known volumes

The existing spherical tests contain 24 fits across two kernel geometries, six deposit designs and two seeds per geometry. On the 24 original truths, the earlier and principal conditional intervals contained the true volume in **21/24 and 21/24** fits, respectively. At 10 cm the principal fit reduced false extension and increased missed fallout in the simulations.

| Area tested | Threshold | Resolved pairs / 24 | False-positive area, earlier → principal (km²) | Missed fallout area, earlier → principal (km²) |
|---|---:|---:|---:|---:|
| Whole Earth | 1 cm | 24 | 214,354 → 206,848 | 4,188 → 4,689 |
| Whole Earth | 10 cm | 24 | 26,557 → 24,427 | 4,640 → 5,056 |
| Whole Earth | 100 cm | 20 | 741 → 739 | 4,726 → 4,727 |
| Gap exposure ≥0.5 | 1 cm | 4 | 7,613 → 1,130 | 0 → 761 |
| Gap exposure ≥0.5 | 10 cm | 4 | 37 → 0 | 11,577 → 13,513 |
| Gap exposure ≥0.5 | 100 cm | 0 | — | — |

Each resolved fit receives equal weight. Both model–truth unions must have at least 64 effective area nodes. Empty or inadequately resolved unions are listed in the CSV. 3 parent fits had sampling warnings and 5 bearing-volume estimates retained split-draw warnings. The scores describe these development simulations.

Four missing-quadrant fits also have a second compatible truth. Within an 80 km disk centred 110 km east and north of the source, a weaker hidden fan gives identical thickness at every sampled site. The tables retain both truths. [Volume errors and failures](known-volumes.csv) · [Contour errors by threshold and truth](known-areas.csv) · [Quadrature replicates and numerical checks](known-deposits.json.gz)
