# YTT · Bayesian nested footprints

The principal bulk volume is **1,994 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **1.37e+03–3.7e+03 km³** across complete posterior draws.

![YTT, Bayesian nested footprints. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The fitted model predicts the four buffered geographic groups with an equal-group negative log score of 1.707. 3 of four independent deletion fits passed their numerical checks.

Each deposit draw expands one or more dispersal footprints. Stations enter through their thickness evidence. The direction and extent of each phase are inferred jointly with the area–thickness curve.

[Joint fit, structural sensitivities and numerical checks](footprints/report.json) · [Held-out predictions](footprints/heldout.csv.gz) · [Contour support and all rejection reasons](footprints/support.json.gz)

[Predicted thickness at audited archive sites](footprints/archive-probes.json)

[Footprint model and assumptions](../../../../docs/methods/nested-footprints.md) · [Matched geography and volume tests](../../../../comparisons/nested-isopachs/geography/README.md) · [Previous corridor maps and original whole-area controls](../../../../archive/2026-11-23-nested-geography/README.md)

[Bulk volume and available thickness evidence in eight compass sectors](footprints/sector-volumes.csv) · [Independent sector integration checks](footprints/sector-checks.json)

## Geographic support and extrapolation

Across draws, the median volume share inside the first measured area rank is 3.9%. The median share outside the last positive rank is 1.1%, with a conditional 95% range of 0.1%–4.7%. A measured site can constrain one part of a perimeter while its opposite side remains unsampled.

The shared geographic screen accepts a complete contour band under 0 of 3 policies. The N, SW sectors contain no accepted off-vent sites. These bearings account for 49.1% of the median-map volume.

[Sector-by-sector evidence and volume](footprints/sector-volumes.csv) · [Volume sensitivity to the priors](footprints/report.json) · [Complete contour decisions](footprints/support.json.gz)
