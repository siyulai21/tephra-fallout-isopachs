# Tephra fallout reconstruction

This repository turns published ash-thickness measurements into **isopachs** (contours of equal thickness) and estimates of bulk fallout volume. Each eruption has one report home with its measurements, source decisions, maps and results.

**[Browse the six eruptions](eruptions/README.md)** · **[Submit a new eruption through chat](docs/NEW_ERUPTION.md)**

## What a case tells you

The **GP ensemble** and **trend-regularized cubic spline** each reconstruct thickness inside the same observation hull. The local support audit shows where observations bracket a contour, including holes inside that hull.

The **directional ensemble** and **spherical mixture** each extend thickness across the Earth under different assumptions. The directional ensemble fits six possible fan structures. Its headline volume integrates the map you see. Its conditional 95% interval comes from integrating complete simulated deposits. The spherical mixture reports its own whole-deposit posterior. A poorly sampled direction can hold ash that none of the observations reveal.

For a worked example, the Minoan case fits the accepted Plinian and co-ignimbrite fallout together. Its case home shows all four volumes and how much ground each covers. [Minoan case](eruptions/minoan/README.md)

## Methods and evidence

[How the four models work](methods/README.md) · [Compare their held-out predictions and stress tests](comparisons/README.md)

To run an analysis or contribute code, start with the [guides and repository layout](docs/README.md). AI agents can follow the [intake, publication and verification instructions](AGENTS.md).
