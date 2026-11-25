# Ilopango TBJ (total primary airfall) · Multiscale directional ensemble

The principal bulk volume is **76.2 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **47.2–133 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.

![Ilopango TBJ (total primary airfall), Multiscale directional ensemble. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 4 cm footprint includes 161 of 168 modeled locations. The thickest displayed line is 1024 cm and encloses 9 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The equal-region negative log score was 1.234. The 95% predictive intervals covered 93.8% of the withheld measurements. 4 of four buffered fits passed the numerical checks.

The directional field allows fan headings and contrast to change with distance. Nearby stations can support local departures. Solid line fragments require exact measurements on both sides and at least 90% posterior probability of a crossing.

The conditional 95% interval for integrated whole-deposit draws is 58.2–524 km³. 0 support policies found a qualifying complete-contour band.

[Fit, structural sensitivities and convergence](multiscale/report.json) · [All contour decisions](multiscale/support.json.gz) · [Measured sites](multiscale/measured-sites.csv.gz) · [Held-out predictions](multiscale/heldout.csv.gz) · [Archive locations](multiscale/archive-probes.json) · [Thickness exceedance at three scales](exceedance.png)

[Previous and revised isopachs at three scales](../../../../comparisons/directional/multiscale/ilopango-tbj.png) · [Matched geographic tests](../../../../comparisons/directional/multiscale/README.md) · [Equations and assumptions](../../../../docs/methods/directional-multiscale.md) · [Previous maps](../../../../archive/2026-11-25-directional-geography/README.md)
