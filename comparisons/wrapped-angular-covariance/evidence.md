# Geographic prediction and geological review

Wrapped angular covariance scored **1.351** on the 36 buffered geographic holdouts. The directional ensemble scored **1.373**, nested footprints **1.465** and spherical mixture **1.500**. Scores average first within a withheld block, then within an eruption. Lower is better. The nine audited principal editions supply 1,179 localities, including 934 point fits, 89 ranges and 156 upper limits. All four methods received identical quantitative evidence and geographic folds. [Paired differences and eruption bootstrap intervals](paired-validation.csv) · [Site-level predictions](events.csv.gz) · [Numerical status of the three controls](current-control-checks.csv)

| Method | Geographic score | 95% coverage at withheld point records |
|---|---:|---:|
| Wrapped angular covariance | 1.351 | 94.9% |
| Directional ensemble | 1.373 | 94.7% |
| Nested footprints | 1.465 | 91.0% |
| Spherical mixture | 1.500 | 91.1% |

The paired difference with directional was −0.023, with a whole-eruption bootstrap interval from −0.093 to +0.066. The intervals for differences with spherical mixture and nested footprints exclude zero in this nine-eruption comparison. The original three-harmonic thinning control scored 1.351 as well. On 35 populated compass holdouts the wrapped model scored 1.524, with 93.99% central 95% coverage. Twenty-nine compass blocks passed numerical checks for all four methods. The empty southwest Millennium quadrant contributes no score. Mean central 95% coverage fell to 88.9% when the outermost quarter of each eruption was withheld. [Fold scores](blocks.csv) · [Compass controls and failures](compass-controls.csv) · [Summary by validation design](summary.csv)

A sensitivity recalibrated preserved-bed probabilities using the other eight eruptions for each target. Its geographic score was 1.342 with 49.5% and 95.2% coverage at nominal 50% and 95%. The uncalibrated forecasts above are the public fit. Calibration has no effect on the geometry or the physical tail. [Leave-one-eruption-out calibration](bed-calibration-summary.csv) · [Training records](bed-calibration-training.csv)

## What the maps show

For each eruption, the four methods use the same observations, contour levels and map windows. The full atlas includes proximal, intermediate and full-deposit views. The links below show the three controls beside the wrapped model. [Nine-eruption wrapped atlas](maps/nine-eruptions.pdf) · [Bearing crossings and reversals](sectors.csv) · [Source-grid refinement](source-topology.csv)

| Eruption | Geological review |
|---|---|
| [Mazama](maps/mazama.jpg) | The northeast fan persists from the proximal to the distal map. At 2 cm the due-west reach shortens from 97 to 78 km relative to three harmonics and its reverse turn falls from 4.5° to less than 0.1°. A118 constrains the local peak. Bolan Lake measures 0.5 cm at 152 km southwest and Little Lake 2 cm at 181 km northwest. A shoulder remains on the thick proximal contours. [Labelled western flank](maps/mazama-western-isopachs.png) |
| [Millennium](maps/b-tm.jpg) | The 2, 8 and 32 cm intermediate outlines turn together into the eastern fan. The former 17.5° southwest reversal at 2 cm disappears. Miy-X helps constrain the thickness south of the vent. Weakening the audited archive response extends the 2 cm southwest crossing from 26.7 to 33.8 km. No quantitative station lies within 22.5° of southwest. [Labelled western flank](maps/b-tm-western-isopachs.png) |
| [Aso-4](maps/aso-4.jpg) | The 2 cm outer line reaches 2,023 km northeast and 938 km south. At 32 cm the inner geometry turns east. The four audited Ulleung cores enter the fit. The 15 cm core at 29.401° N, 137.347° E is 709 km away at 121°, east-southeast. The model gives it 10.2 cm in training and 4.9 cm in geographic holdout. Far-south reach remains sparsely constrained. |
| [Youngest Toba Tuff](maps/ytt.jpg) | At 225° the 32, 16 and 8 cm curves cross at 500, 679 and 910 km without a station-scale notch. The 2 cm line reaches 1,546 km. The source-prior integral resolves one vent-enclosing 512 cm contour. Seven additional southwest ash-bearing cores inform occurrence, with mixing and erosion limiting thickness interpretation. |
| [Ilopango TBJ](maps/ilopango-tbj.jpg) | The 2 cm contour reaches 135 km at 15° and 426 km due south. The broad intermediate northern flank overpredicts the 0.35 cm Petapilla bed, giving 2.67 cm in the full fit and 14.16 cm with its geographic block withheld. This is a failure at a measured site despite a smooth outline. |
| [Minoan](maps/minoan.jpg) | Western 4 and 8 cm boundaries are gradual. The thick intermediate maximum lies east of the vent, with a separate local near-source high. The directional control has stronger local indentations and nested footprints extend farther east. |
| [Pinatubo layer C](maps/pinatubo-1991-layer-c.jpg) | The west-southwest intermediate fan broadens into the distal footprint. A small off-vent 16 cm feature southwest of the source persists under source-grid refinement. |
| [Sakurajima–Satsuma](maps/sakurajima-satsuma.jpg) | Rounded inner contours acquire a southeast component through the intermediate and outer scales. Local undulations are stronger in the spherical mixture. The older Aira caldera has no event-specific role in this fit. |
| [Tambora](maps/tambora.jpg) | The intermediate outlines shift laterally and the distal outlines close gradually. With 26 stations, similar-looking central contours permit a wide range of distal areas in the posterior. |

Positive occurrences, independently inspected cores and explicit quantitative limits carry different information. The public archive comparisons record site-specific predictions, enclosure probabilities and overlap with the quantitative table. They include the Mazama lakes, Aso marine cores, southwest Indian Basin and Ilopango sites. [Archive site predictions](archives/) · [Sites by sector](fan-sector-stations.csv) · [Angular uncertainty](uncertainty.csv)

## How much volume can escape a survey?

In a frozen deposit with a true volume of 120 km³ and a diffuse tail, the local network gave a coherent-volume median of **42.8 km³**, with a central 95% interval of **30.1–104.9 km³**. Adding distant cores gave **124.4 km³** and **113.2–137.0 km³**. A different frozen network supplies identical observations for two hidden-sector truths of 33.6 and 72.8 km³. Both experiments show why a convincing contour through sampled ground cannot identify a remote hidden fan. Two independent equal-area integrations and 12 frozen networks also test turning, overlapping and narrow fans. [Known-volume and area recovery](known-volume.csv) · [Frozen observations and truth grids](known-inputs/)

For Toba, the pointwise median map integrates to **2,056 km³**. The pointwise quartile maps integrate to **1,278 and 3,352 km³**. The central 95% interval of whole-field volumes is much wider, **1,313–16,111 km³**, after refinement to two independent populations of 16,384 coherent draws. A broad region can thicken together in one realization. [Toba spatial and sampling refinements](../../eruptions/ytt/results/wrapped-angular-covariance/earth-integrals.json)

The principal estimate integrates the pointwise median of the entire fitted field. The study also anchors segmented exponential, power-law and Weibull continuations at geographically screened mapped contours and tests them against withheld distal beds and independent published maps. [Thickness–area results](thickness-area/README.md) · [Reproduction commands](../../benchmarks/wrapped_angular/README.md)
