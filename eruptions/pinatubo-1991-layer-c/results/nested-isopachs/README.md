# Pinatubo 1991 (Layer C) · Bayesian nested footprints

**November 24 update.** The current principal estimate is 7.37 km³ on 235 sites. The figures below document the 208-site comparison. [Current nested results](2026-11-24/README.md) · [Data re-audit](../../../../data/audits/Pinatubo-1991/reaudit/README.md)

The principal bulk volume is **6.7 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **5.01–15 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth. **Numerical warning on the fitted deposit distribution.**

![Pinatubo 1991 (Layer C), Bayesian nested footprints. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 191 of 208 modeled locations. The thickest displayed line is 16 cm and encloses 40 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The fitted model predicts the four buffered geographic groups with an equal-group negative log score of 1.979. 3 of four independent deletion fits passed their numerical checks.

Each deposit draw expands one or more dispersal footprints. Stations enter through their thickness evidence. The direction and extent of each phase are inferred jointly with the area–thickness curve.

[Joint fit, structural sensitivities and numerical checks](footprints/report.json) · [Held-out predictions](footprints/heldout.csv.gz) · [Contour support and all rejection reasons](footprints/support.json.gz) · [Archive-site predictions](footprints/archive-probes.json) · [Volume and observations by compass sector](footprints/sector-volumes.csv)

[Footprint model and assumptions](../../../../docs/methods/nested-footprints.md) · [Matched geography and volume tests](../../../../comparisons/nested-isopachs/geography/README.md) · [Previous corridor maps and original whole-area controls](../../../../archive/2026-11-23-nested-geography/README.md)

[Bulk volume and available thickness evidence in eight compass sectors](footprints/sector-volumes.csv) · [Independent sector integration checks](footprints/sector-checks.json)

## Geographic support and extrapolation

Across draws, the median volume share inside the first measured area rank is 0.8%. The median share outside the last positive rank is 14.6%, with a conditional 95% range of 1.4%–78.7%. A measured site can constrain one part of a perimeter while its opposite side remains unsampled.

The shared geographic screen accepts a complete contour band under 0 of 3 policies.

[Sector-by-sector evidence and volume](footprints/sector-volumes.csv) · [Volume sensitivity to the priors](footprints/report.json) · [Complete contour decisions](footprints/support.json.gz)
