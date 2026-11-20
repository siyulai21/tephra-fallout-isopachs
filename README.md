# Tephra fallout reconstruction

This repository turns published ash-thickness measurements into **isopachs** (contours of equal thickness) and estimates of bulk fallout volume. Open an eruption to see all four fits on one map panel, their volumes and their predictions at withheld sites. Its case page also links the measurements and source decisions.

**[Browse the six eruptions](eruptions/README.md)** · **[Submit or recalculate an eruption through chat](docs/NEW_ERUPTION.md)**

## What a case tells you

The **GP ensemble** and **trend-regularized cubic spline** each reconstruct thickness inside the same observation hull. The local support audit shows where observations bracket a contour, including holes inside that hull.

The **directional ensemble** and **spherical mixture** each extend thickness across the Earth under different assumptions. The principal directional map allows a fan to turn in poorly sampled directions. Its bulk volume integrates the pointwise median thickness over the Earth. Its conditional 95% interval comes from integrating complete simulated deposits. The spherical mixture reports its own whole-deposit posterior. A poorly sampled direction can hold ash that none of the observations reveal.

On the directional maps, solid contour fragments connect measured thicknesses that bracket a level. Dashes show where the deposit edge depends on continuation. Each directional report compares the principal map with a control that fixes fan direction using the same observations and weights. [Six-eruption comparison and maps](comparisons/directional/README.md)

Minoan shows how the Plinian and co-ignimbrite fans can be fitted as one deposit. B-Tm shows how glass-shard profiles and a peat mass budget constrain the reach of fallout contours. The case compares the resulting maps with the original thickness survey and two weaker archive interpretations. [Minoan case](eruptions/minoan/README.md) · [B-Tm case](eruptions/b-tm/README.md)

## Methods and evidence

[How the four models work](methods/README.md) · [Compare their held-out predictions and stress tests](comparisons/README.md)

To run an analysis or contribute code, start with the [guides and repository layout](docs/README.md). AI agents can follow the [intake, publication and verification instructions](AGENTS.md).
