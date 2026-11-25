# Tephra fallout reconstruction

Fit isopachs to measured ash thicknesses and documented bounds, then integrate each map to estimate bulk airborne fallout. Open an eruption to see the fitted maps together, compare volumes and inspect the evidence behind them.

**[Eruption catalogue](eruptions/README.md)** · **[Submit eruption data in chat](docs/NEW_ERUPTION.md)** · **[Download the results](comparisons/principal-volumes.csv)**

The GP ensemble and trend-regularized cubic spline integrate over the same observation hull. The directional ensemble, spherical mixture and Bayesian nested dispersal footprints extend over Earth. For each method the centre estimate integrates its median-thickness map. The conditional range integrates the 25th- and 75th-percentile thickness maps over the same area. Geographic holdouts test prediction, and a separate support audit identifies contour fragments bracketed by observations. [Directional isopachs at three scales](comparisons/directional/multiscale/README.md) · [How to read the figures and volumes](docs/OUTPUTS.md) · [Methods and validation](comparisons/README.md)

Each method fits a site reported only as a lower limit as a measurement at that thickness, at every distance and bearing. The fitting table preserves the original qualifier. Upper limits retain an upper-censored likelihood and ranges use both endpoints. The seven affected eruptions include 264 reported minima. [Changes by eruption and distance](comparisons/reported-minima/README.md) · [Fitting and plotting rules](docs/methods/reported-minimum-fitting.md)

For a new eruption, preserve the source records and digitize only evidence missing from the tables. The intake guide covers measurements, bounds, thicknessless occurrences, map registration and publication. [Intake workflow](docs/NEW_ERUPTION.md) · [Agent instructions](AGENTS.md) · [Code and method guides](docs/README.md)
