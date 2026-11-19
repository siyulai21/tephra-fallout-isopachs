# B-Tm fallout reconstruction

The revised input contains **64 thickness measurements, three lower bounds and one upper bound**. All four methods target combined airborne fallout from both phases of the 946 CE Millennium eruption. [Accepted thickness CSV](../../data/eruptions/B-Tm_reviewed.csv) · [Source audit](../../data/audits/B-Tm/README.md)

## Directional reach and archive evidence

The measured-fan median map integrates to **42.5 km³**. The complete-deposit median is 52.3 km³, with a conditional 95% interval of 28.0–120.8 km³. The map shows where the predicted extent of each level changes under the recorded assumptions.

The archive sensitivity **Comparable visible fallout beds** gives a median-map volume of **35.8 km³**. The complete-deposit median is 43.5 km³ (conditional 95% interval 24.8–70.4 km³). This result depends on the stated transfer between local fallout and a recognizable layer.

![Agreement on median isopach extent](results/directional/bearing/isopach-agreement.png)

[Directional estimates, inspected archives and transfer sensitivities](results/directional/bearing/README.md)

## Four matched thickness-only fits

The directional control in this four-panel comparison includes a possible fan in the largest sampling gap. Its scores and volumes refer to that six-structure model.

![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at B-Tm](results/four-method-isopachs.png)

All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. The regional fits stop there. The global fits also show how thickness continues into unsampled ground. Gaps inside the hull can still be poorly constrained.

## Volume and predictive performance

| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |
|---|---|---:|---:|---:|
| GP ensemble | Hull mean **24.33** | 15.66–37.07 | 1.512 | 89.1% |
| Cubic spline | Hull mean **28.47** | 15.52–66.18 | 1.803 | 98.4% |
| [Directional ensemble](results/directional/README.md) | Whole-Earth median map **54.0** · volume median 73.5 | 31.8–1,211.9 | 1.304 † | 93.8% |
| [Spherical mixture](results/spherical-mixture/README.md) | Whole-Earth volume median **158.1** | 40.7–4,461.6 | 1.490 † | 93.8% |

A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out observations inside the 95% predictive interval. † A geographic fit had a numerical warning or incomplete scoring.

The directional map integrates pointwise median thickness. Measured interpolation triangles account for **8.2%** of its volume. The directional volume interval comes from complete simulated deposits. The regional intervals condition on fitted settings. [Regional maps, fits and diagnostics](results/README.md)

<details>
<summary>Compare the two whole-Earth maps</summary>

| Directional ensemble | Spherical mixture |
|---|---|
| ![Directional ensemble median thickness over the Earth for B-Tm](results/directional/ensemble/world.png) | ![Spherical-mixture median thickness over the Earth for B-Tm](results/spherical-mixture/world.png) |

Both maps use the vent longitude as their central meridian. Read the thickness key under each map.

</details>


## What changed in the data

The previous calculation fitted 84 scalars. The audit recovered field thicknesses in four Japanese lake cores and six named proximal sections. It recoded Lake Suigetsu as a sub-millimetre upper bound. West of the vent, Sihailongwan and Quanyang establish the presence of ash, with insufficient information to estimate a physical thickness. Costa’s Lake Kushu coordinate was about 314 km from the lake. A measured bed thickness could not be verified there. [Source audit and evidence map](../../data/audits/B-Tm/README.md) · [Original input and case history](history/2026-11-18/README.md)

Reintroducing the twelve numerical ranges at their low and high endpoints gives median-map integrals of 50.8 and 59.2 km³. Assigning the two disputed proximal stacks to Millennium fall gives 46.8 km³. [Source alternatives and numerical checks](results/source-audit/README.md)

The 1993 Pyongyang edition of *Geology of Korea* records 7 m of pumice at Samjiyon. Its stratigraphy and its chapter on volcanism distinguish bed thickness from clast size. A separate field party confirmed metre-scale pumice in the area in 2002. The exact outcrop, facies and contribution of both phases remain uncertain. Treating 7 m as the combined fallout gives a median-map integral of 63.9 km³. The lower-limit reading gives 66.6 km³ and the complete-stack upper-limit reading gives 53.6 km³. The earlier report of 1.6 m around Samjiyon yields 57.3 km³ under the same total-fall assumption. Each alternative has its own conditional interval. [Samjiyon field evidence](../../data/audits/B-Tm/samjiyon.md) · [Intervals and predictions at Samjiyon](results/source-audit/README.md)

With the previous 84 scalar records, the directional median-map integral was 59.3 km³. The audited selection gives 54.0 km³. The spherical posterior median changes from 106.3 to 158.1 km³. These changes reflect the entire source revision. The regional integration hull and holdout groups also change with the observations. [Previous release](history/2026-11-18/README.md)

## Where the estimates are supported

The standard interpolation footprint covers 96,131 km². Of the twelve checks across four methods and three support policies, 0 identified a complete band of supported isopachs. Global totals depend on assumptions about unsampled directions, near-vent deposition and the distal tail. [Regional support](results/support/README.md) · [Directional contour decisions](results/directional/ensemble/complete-contour-audit.json) · [Spherical contour decisions](results/spherical-mixture/complete-contour-audit.json)

The directional median-map numerical screen passed. 6 of six full-data structures passed every sampling and integral diagnostic. The geographic comparison scored 68 of 68 records, including likelihood bounds. Geographic folds with sampling warnings include 0. Of four source-sector holdouts, 2 had no observations. Sampling warnings remained in 1 of the 2 estimable holdouts. The spherical baseline diagnostic passed. Spherical prior refits with outstanding numerical warnings include wide geography, five fans. Each family report retains its fold diagnostics. [Directional ensemble](results/directional/README.md) · [Spherical mixture](results/spherical-mixture/README.md)

## Does segmented thinning explain Samjiyon?

None of the disputed DPRK locality summaries enters the principal fit. The transect through the workbook position is almost entirely supplied by the extrapolated radial curve. Across the explicit-break refits, the predicted probability above 7 m is 0.27%–0.28% at that location. The break radii remain poorly constrained. [Matched thinning-rate experiment and geographic warnings](results/source-audit/thinning/README.md) · [Which near-vent sections enter the principal estimate](../../data/audits/B-Tm/README.md)

In the matched single-structure calculation, the smooth median map integrates to 48.4 km³. Adding one shared radial break gives 48.4 km³ and adding two gives 48.6 km³. These are controlled changes to the thinning law at fixed fan geometry. [Whole-deposit intervals, median maps and numerical diagnostics](results/source-audit/thinning/volumes/README.md).

Omitting the 26 thickness values carried from compilations leaves 42 observations. With the same six ensemble structures, the median-map integral becomes 48.9 km³. The predicted local median at the withheld workbook point is 49.7 cm, with probability 0.19% above 7 m. [Which records were removed](../../data/audits/B-Tm/source-class-provenance.json) · [Conditional intervals and full local draws](results/source-audit/README.md)
