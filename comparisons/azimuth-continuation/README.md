# How far should isopachs extend into unsampled directions?

The thickness-only ensemble can carry a broad contour west of Changbaishan or southwest of Mazama without local thickness measurements. In the revised maps, quantitatively bracketed pieces of the median contour are solid. A dashed closure shows where a model carries that contour through a gap.

The existing ensemble anchors a thick deposit at the vent. Its broader fans can carry that thickness sideways, and three of its six structures place a possible fan in the largest distal sampling gap. The centre of that hypothetical fan is determined by the survey geometry. The principal uncertain-azimuth ensemble varies the direction of extra fallout coherently across a whole deposit. This shortens several median contours and allows individual draws to reach a withheld direction. The project uses its quantitative median isopachs and their whole-Earth integral as the principal directional reconstruction. The fixed-direction control measures how much the assumption matters.

## The two reported examples

**Western Changbaishan.** At Lake Sihailongwan the preceding fixed-direction map gave 15.8 cm. The paired fixed-direction control gives 16.4 cm and the uncertain-azimuth median gives 8.0 cm. The principal quantitative-only 95% latent range is 0.00015–101 cm. A cryptotephra occurrence without a measured fallout bed still challenges this central prediction. The core description supplies no numerical upper limit. [Primary core study](https://doi.org/10.1016/j.gr.2015.01.013) · [Source and site audit](../../eruptions/b-tm/results/directional/bearing/field-evidence.md)

Conditioning only on an assumed 1 cm recognizable bed at Quanyang gives a Sihailongwan median of 0.0084 cm without conditioning on the lake. Assuming recognizable beds at 1 cm at Sihailongwan and 2 cm at Quanyang gives 0.0052 cm at the lake and 0.11 cm at the peat. Those estimates depend on preservation and detection. The second estimate uses both sites in its likelihood. The fixed record keeps transferred shard-to-thickness conversions outside the quantitative table. [B-Tm maps and all archive sensitivities](../../eruptions/b-tm/results/directional/azimuth/README.md)

**Southwestern Mazama.** Along the 225° ray the 10 cm median crossing moves from 199 to 135 km. The ≥10 cm area across the entire southwest octant falls from 13,559 to 6,635 km². The accepted observations do not locate an actual southwest deposit edge. [Mazama maps and all eight bearings](../../eruptions/mazama/results/directional/azimuth/README.md)

## Global bulk volume

The two median maps use identical structure weights and training observations. Their difference measures the effect of uncertain fan direction. Earlier case headlines used their original calibration. Every figure below integrates to the antipode.

| Eruption | Principal uncertain-direction median map (km³) | Fixed-direction control (km³) | Uncertain-azimuth complete deposits, median [95%] (km³) |
|---|---:|---:|---:|
| [Aso-4](../../eruptions/aso-4/results/directional/azimuth/README.md) | **659.1** | 779.7 | 916.3 [297.9–3,511.5] |
| [B-Tm](../../eruptions/b-tm/results/directional/azimuth/README.md) | **44.1** | 56.2 | 72.9 [31.0–1,507.8] |
| [Mazama](../../eruptions/mazama/results/directional/azimuth/README.md) | **166.1** | 177.0 | 213.5 [138.6–1,265.1] |
| [Minoan](../../eruptions/minoan/results/directional/azimuth/README.md) | **28.8** | 32.9 | 39.1 [21.3–138.5] |
| [Tambora](../../eruptions/tambora/results/directional/azimuth/README.md) | **95.7** | 107.7 | 117.5 [73.2–203.8] |
| [YTT](../../eruptions/ytt/results/directional/azimuth/README.md) | **900.6** | 1,005.6 | 1,272.1 [736.5–3,660.9] |

## Do the shorter contours predict withheld observations?

Each eruption learns mixture weights from the other eruptions. Both alternatives are scored on exactly the same withheld rows and limits. Lower negative log predictive scores are better. Averages give each available fold equal weight, then each eruption equal weight.

| Deletion design | Eruptions contributing | Fixed | Uncertain azimuth |
|---|---:|---:|---:|
| Geographic, all estimable folds | 6 | 1.602 | 1.597 |
| Geographic, passing folds only | 6 | 1.777 | 1.706 |
| Quadrants, all estimable folds | 6 | 1.939 | 2.007 |
| Quadrants, passing folds only | 5 | 1.854 | 1.912 |

The uncertain-azimuth model improves the pooled geographic score slightly. Its quadrant score worsens, including in the passing-only comparison. The primary reporting choice favors less extension of the median contour in unsupported directions. YTT benefits in both designs. Holding out the target protects calibration weights. The method itself was developed with these six eruptions in view. Tambora has no passing quadrant refit. Some other individual fold fits also carry numerical warnings. The fixed-direction results remain available as sensitivities. [Case scores, contributing folds and volumes](case-comparison.csv)

## Known deposits with a deliberate sampling gap

Fourteen previously specified simulated datasets cover seven designs and two seeds. Two compatible truths on each invisible-sector dataset share all observations but differ by a hidden lobe. The table evaluates false extension and missed fallout on area-weighted whole-Earth nodes, separately for locations with directional exposure of at least 0.5. Each design and seed receives equal weight; compatible truths share that weight. Cells require at least 64 effective union nodes for each of the three mapped decisions.

| Region | Thickness | Resolved datasets | False extension, fixed → uncertain (km²) | Missed area, fixed → uncertain (km²) | Overlap, fixed → uncertain |
|---|---:|---:|---:|---:|---:|
| whole Earth | 1 cm | 14 | 160,917 → 153,346 | 3,538 → 3,861 | 0.797 → 0.805 |
| whole Earth | 10 cm | 14 | 15,000 → 14,022 | 11,646 → 11,796 | 0.787 → 0.788 |
| whole Earth | 100 cm | 12 | 80 → 81 | 6,674 → 6,674 | 0.109 → 0.109 |
| high exposure | 1 cm | 4 | 1,627 → 112 | 0 → 690 | 0.953 → 0.976 |
| high exposure | 10 cm | 4 | 0 → 0 | 6,054 → 6,367 | 0.619 → 0.589 |
| high exposure | 100 cm | 4 | 0 → 0 | 6,228 → 6,228 | 0.000 → 0.000 |

Across the resolved high-exposure tests, the revised 1 cm contour cuts false extension at the cost of some missed area. For 10 cm there was already no false extension in those tests, and shortening the contour misses more true fallout. The stricter 90% exceedance outline misses still more. Nine of the fourteen synthetic datasets had a sampling warning in at least one base structure. All four datasets with resolved high-exposure 1 cm and 10 cm comparisons had such a warning. The passing-only subset therefore provides no resolved test of those high-exposure comparisons. These patterns describe a trade-off under the simulated assumptions. They cannot establish the unseen shape of a real deposit. [False-positive and false-negative areas for every trial and threshold](synthetic-areas.csv) · [True volumes and base-fit warnings](synthetic-trials.json)

## Numerical checks and regional comparison

The spatial quadrature check passed in all six eruptions. For B-Tm the 100 cm area differs by 24% between split posterior packets at the 512-draw-per-chain ceiling, although the integrated median-map volume differs by about 1%. The thickest contour therefore remains numerically sensitive. Passing integration checks do not identify ash volume outside the measurements.

The matched regional benchmark was rerun on the same six audited inputs. Its twelve empirical rows and its eighty synthetic rows reproduced the published CSVs byte for byte, including the four unsuccessful synthetic fits. [Matched regional results](../../eruptions/b-tm/results/source-audit/matched-regional.csv) · [Regional synthetic results](../regional/synthetic/comparison.csv)

[Mathematical method, map rules and reproduction commands](../../docs/methods/uncertain-azimuth.md) · [Previous measured-fan comparison](../bearing-continuation/README.md) · [Previous primary maps and reports](archive/2026-11-18-before-primary/README.md) · [Earlier report pages](archive/2026-11-18/README.md)
