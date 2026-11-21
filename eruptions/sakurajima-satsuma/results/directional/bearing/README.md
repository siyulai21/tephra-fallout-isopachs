# Sakurajima Satsuma (P14) directional continuation

The median of the ensemble using measured fan directions integrates to **8.2 km³ of bulk fallout**. Angular reach varies across extrapolation assumptions. Changing reach alone preserves the field inside measured triangles. Conditioning on archive evidence can update the fitted thicknesses. A hypothetical extra fan in the largest sampling gap remains a separate structural sensitivity.

![Agreement and disagreement about the median isopachs](isopach-agreement.png)

Each panel compares the same thickness level across the mapped scenarios. Ochre marks locations where changing the assumptions changes which side of an isopach the site occupies. The dashed outline encloses the measured interpolation footprint. The diamond marks an inspected archive where one is available. A scenario median below a level still permits ash at that site.

![Conditional median isopachs on a common scale](conditional-isopachs.png)

![Median isopachs across the complete survey region](median-deposit.png)

The colour scale is shared across panels. Values below 0.01 cm share the lowest colour. Diamonds mark inspected archives. The full-deposit integrals extend beyond the displayed survey area through the antipode.

![Median isopach segments bracketed by measured sites](measured-fragments.png)

Each segment lies inside an accepted interpolation triangle with measurements on both sides of the level and at least 90% conditional probability that the level crosses its nodes. Gaps and disconnected patches remain. Candidate narrow rays in the full-field panels can extend beyond the station that fixes their bearing. Only the segments here passed the local bracketing test.

| Continuation | Integral of median map (km³) | Median complete deposit (conditional 95%, km³) | Numerical checks |
|---|---:|---:|---|
| Measured fan directions | 8.17 | 9.9 (4.3–73.4) | Passed |

### Bulk volume of the median map by direction

Each entry is km³ in a 45° sector centred on the named bearing from the vent.

| Continuation | N | NE | E | SE | S | SW | W | NW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Measured fan directions | 0.466 | 0.315 | 0.718 | 2 | 2.95 | 0.958 | 0.486 | 0.278 |

The sector volumes sum through the antipode and include extrapolated proximal, unsampled and distal fallout. The conditional intervals integrate individual deposits. The sector draws are retained for each fitted deposit. Agreement among medians cannot establish an observation-supported closed band.

**Measured fan directions** scored geographic 1.244, quadrants 1.693. Geographic sampling warnings occurred in zero folds. The shared screen accepted 0 complete bands across its policies. [Geographic and directional folds](quantitative/validation.json) · [Contour decisions](quantitative/complete-contour-audit.json)

## Broad spherical continuation control

This comparison uses the original unattenuated spherical draws. The principal spherical case integrates a range of possible continuations across unobserved bearings.

These archive predictions use the spherical posterior fitted to the quantitative measurements and bounds. Each complete deposit retains its original weight.

| Archive transfer | Complete-deposit median (conditional 95%, km³) | Weight ESS | Numerical check |
|---|---:|---:|---|
| Principal observation bounds and thicknesses | 14.3 (6.3–921.5) | 6,000 | Passed |

[Spherical archive probabilities, chain checks and hull membership](spherical-archive-checks.json) · [Posterior weights by chain](spherical-archive-weights.npz)

## Reproducible results

- [Measured fan directions](quantitative/result.json) · [Median field packet](quantitative/map-fields.npz) · [Complete-field sector draws](quantitative/sector-draws.npz)

[Map design](map-design.json) · [Numerical map grids](grids-log10-cm.npz)
