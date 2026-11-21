# Tephra fallout reconstruction

This repository turns published ash-thickness measurements into **isopachs** (contours of equal thickness) and estimates of bulk fallout volume. Open an eruption to compare five fits, their volumes and their predictions at withheld sites. Its case page also links the measurements and source decisions.

**[Browse the eruption results](eruptions/README.md)** · **[Submit or recalculate an eruption through chat](docs/NEW_ERUPTION.md)**

## What a case tells you

The **GP ensemble** and **trend-regularized cubic spline** each reconstruct thickness inside the same observation hull. The local support audit shows where observations bracket a contour, including holes inside that hull.

The **directional ensemble** and **spherical mixture** each extend thickness across the Earth under different assumptions. The principal directional map allows a fan to turn in poorly sampled directions. Its bulk volume integrates the pointwise median thickness over the Earth. Its conditional 95% interval comes from integrating complete simulated deposits. The spherical mixture limits how far a measured bearing supports broad fallout and allows an unseen fan to turn. Its principal volume is the median integral of complete deposits. A poorly sampled direction can hold ash that none of the observations reveal. [Spherical maps and matched tests](comparisons/spherical-mixture/README.md)

On the directional maps, solid contour fragments connect measured thicknesses that bracket a level. Dashes show where the deposit edge depends on continuation. Each directional report compares the principal map with a control that fixes fan direction using the same observations and weights. [Cross-eruption comparison and maps](comparisons/directional/README.md)

Minoan combines Plinian and co-ignimbrite fans in one deposit. At B-Tm, two lake profiles and one peat core supplied conditional upper bounds where no bed thickness was measured. Their case pages show how those decisions affect the maps and volumes. [Minoan](eruptions/minoan/README.md) · [B-Tm](eruptions/b-tm/README.md) · [How to review an occurrence without a measured thickness](docs/methods/sites-without-measured-thickness.md)

The **Bayesian nested-isopach model** estimates the area covered above every thickness threshold. A deposit can have multiple fans or a detached maximum. Each case shows the median isopachs, the volume in each thickness range and the fraction inferred beyond the farthest exact site. [Compare all three global estimates](comparisons/nested-isopachs/README.md)

## Methods and evidence

[How the five models work](docs/methods/README.md) · [Compare their held-out predictions and stress tests](comparisons/README.md)

To run an analysis or contribute code, start with the [guides and repository layout](docs/README.md). AI agents can follow the [intake, publication and verification instructions](AGENTS.md).
