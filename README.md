# Tephra fallout reconstruction

This repository reconstructs ash thickness and bulk airborne fallout volume from field measurements and documented bounds. Open an eruption to compare five independent fits, see where they draw each isopach and check their predictions against geographically withheld sites.

**[Explore the eruption catalogue](eruptions/README.md)** · **[Submit an eruption in chat](docs/NEW_ERUPTION.md)**

The latest model comparison traces the Pinatubo, Ilopango, Aso-4, Mazama and Tambora anomalies and tests the changes across all nine eruptions. [Maps, held-out predictions and uncertainty](comparisons/global-extent/README.md)

The accepted Mazama dataset contains 300 localities, including 29 two-sided thickness intervals. The published maps and volumes use the preceding 271-location edition. [Current field audit and principal CSV](data/audits/Mazama/principal-ranges/README.md) · [Published Mazama results](eruptions/mazama/README.md)

## Read a result

The **GP ensemble** and **trend-regularized cubic spline** map the same observation hull. The **directional ensemble**, **spherical mixture** and **Bayesian nested footprints** extend thickness across Earth under different assumptions about unsampled ground. Each global method reports one median-thickness map and the integral of that same field over Earth. Conditional 95% intervals describe uncertainty in total fallout volume. The regional maps integrate median thickness within the observation hull. [All five methods](docs/methods/README.md) · [Numerical results across eruptions](comparisons/principal-volumes.csv)

Each global figure shows the full modeled footprint down to 0.5 cm, a near-vent view and thickness versus the square root of cumulative area. Contours and observation colours follow 0.5, 1, 2, 4 cm and successive doublings. The global volumes include thinner fallout. A case’s support audit shows which contour fragments have local measurement brackets and where continuation relies on assumptions. [Map and volume definitions](docs/OUTPUTS.md)

Minoan combines Plinian and co-ignimbrite fallout in one deposit. B-Tm includes interpreted upper bounds from lake and peat archives. Pinatubo combines the 15 June 1991 climactic land deposit with its correlated marine ash. The case links each mapped reading to its original pixels and each accepted core to its source. [Minoan](eruptions/minoan/README.md) · [B-Tm](eruptions/b-tm/README.md) · [Pinatubo Layer C](eruptions/pinatubo-1991-layer-c/README.md)

[Prediction tests and sensitivity studies](comparisons/README.md) · [Repository and code guides](docs/README.md) · [Agent workflow](AGENTS.md)
