# Where do observations support a contour?

Each eruption has one local support audit. It removes long bridges between sampling sites, preserves holes in the footprint and checks whether measured thicknesses bracket a contour. A supported patch can give a partial volume even when the measurements do not establish a complete band of isopachs. [Support protocol](../../docs/methods/supported-isopachs.md)

| Eruption | Local footprint and regional partial volumes | Complete-contour decisions for the global fits |
|---|---|---|
| Aso-4 | [GP and cubic spline](../../eruptions/aso-4/results/support/README.md) | [Directional ensemble](../../eruptions/aso-4/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/aso-4/results/spherical-mixture/support/) |
| B-Tm | [GP and cubic spline](../../eruptions/b-tm/results/support/README.md) | [Directional ensemble](../../eruptions/b-tm/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/b-tm/results/spherical-mixture/support/) |
| Ilopango TBJ, Unit G | [GP and cubic spline](../../eruptions/ilopango-tbj/results/support/README.md) | [Directional ensemble](../../eruptions/ilopango-tbj/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/ilopango-tbj/results/spherical-mixture/support/) |
| Mazama | [GP and cubic spline](../../eruptions/mazama/results/support/README.md) | [Directional ensemble](../../eruptions/mazama/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/mazama/results/spherical-mixture/support/) |
| Minoan | [GP and cubic spline](../../eruptions/minoan/results/support/README.md) | [Directional ensemble](../../eruptions/minoan/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/minoan/results/spherical-mixture/support/) |
| Pinatubo 1991, Layer C | [GP and cubic spline](../../eruptions/pinatubo-1991-layer-c/results/support/README.md) | [Directional ensemble](../../eruptions/pinatubo-1991-layer-c/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/pinatubo-1991-layer-c/results/spherical-mixture/support/) · [Nested corridors](../../eruptions/pinatubo-1991-layer-c/results/nested-isopachs/support.json) |
| Sakurajima Satsuma | [GP and cubic spline](../../eruptions/sakurajima-satsuma/results/support/README.md) | [Directional ensemble](../../eruptions/sakurajima-satsuma/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/sakurajima-satsuma/results/spherical-mixture/support/) |
| Tambora | [GP and cubic spline](../../eruptions/tambora/results/support/README.md) | [Directional ensemble](../../eruptions/tambora/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/tambora/results/spherical-mixture/support/) |
| YTT | [GP and cubic spline](../../eruptions/ytt/results/support/README.md) | [Directional ensemble](../../eruptions/ytt/results/directional/azimuth/complete-contour-audit.json) · [Spherical mixture](../../eruptions/ytt/results/spherical-mixture/support/) |

For B-Tm, the three interpreted limits enter the likelihood and enlarge the observation hull. The standard footprint still uses the locations of exact thickness measurements. Comparing partial volumes on that footprint answers a different question from integrating a conditional deposit across the Earth. [B-Tm footprint and contours](../../eruptions/b-tm/results/support/README.md) · [Effect of the three cores](../../eruptions/b-tm/results/cryptotephra/README.md)

The spherical support folders pair the compressed decisions with the exact posterior field samples used to test contours. The directional links give each contour decision as readable JSON.

## What the design studies tested

The synthetic trials check whether the screen rejects contours crossing a gap or encircling a hidden lobe. Their reports retain all trials, including rejection reasons and cases where coverage is poor. The separate padding experiment retuned the spline on four fitting rectangles and integrated the same footprint. Each study records its original selection. [Known-deposit support trials](stress-tests/README.md) · [Fitting-rectangle sensitivity](fitting-domain/README.md) · [Original fit-parity checks](validation.json)
