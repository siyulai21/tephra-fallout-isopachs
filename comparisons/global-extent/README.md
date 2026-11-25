# Global fallout models and geographic validation

## Finite-spread validation on 300 Mazama localities

The directional, spherical and corridor pairs each share the same data, geographic holdouts and parent fields. The nested corridor is a comparator for this development experiment. The current nested fit uses dispersal footprints. [Five current maps and integrated thickness bands](../../eruptions/mazama/README.md) · [Paired predictions](paired-results.csv) · [Parent map integrals](mazama-parent-checks.json)

| Family | Parent median map (km³) | With finite spread (km³) | Parent geographic score ↓ | With finite spread ↓ | Passing folds |
|---|---:|---:|---:|---:|---:|
| Directional ensemble | 280.0 | 277.5 | 0.786 | 0.739 | 2/4 |
| Spherical mixture | 429.5 | 417.7 | 1.053 | 1.027 | 4/4 |
| Nested corridor control | 257.5 | 241.2 | 0.757 | 0.820 | 2/4 |

The independent spherical-kernel integral passed. [Kernel checks](kernel-integrals.json)

## Across the nine study inputs

The finite-spread experiment uses the 208-site Pinatubo Layer C and 56-site Ilopango Unit G editions. [Matched inputs and results](../../archive/2026-11-24-case-editions/README.md)

The paired study measures how limiting lateral spread away from observations changes isopach area, integrated volume and prediction in a withheld region. Each eruption contributes one geographic score to the average. Lower scores indicate better predictions. [All 27 pairs](paired-results.csv) · [Sector areas and volumes](sector-areas.csv.gz)

| Family | Mean parent score | Mean finite-spread score | Improved cases |
|---|---:|---:|---:|
| Directional ensemble | 1.487 | 1.458 | 9/9 |
| Spherical mixture | 1.387 | 1.408 | 3/9 |
| Nested corridor control | 1.554 | 1.607 | 2/9 |

![Parent and finite-spread contours on matched axes for five eruptions](affected-contours.png)

Each pair uses the same observations, axis limits and posterior subset. Sites are colored by thickness bracket. The Mazama corridor pair uses all 300 current localities. [Angular variation and input hashes](angular-variation.csv)

The independent survey tests compare predictions at field sites withheld from training. The simulated deposits measure both excess mapped area and missed area against known contours. [External field surveys](external-surveys.csv) · [Known-deposit trials](known-deposits.csv.gz) · [Finite-spread method](../../docs/methods/finite-spread.md)
