# How well do the methods predict fallout?

The catalogue contains 43 fits on the current inputs for nine eruptions. Eight eruptions have all five methods, including Pinatubo on 235 Layer C localities. Ilopango has three global fits on 168 all-phase airfall localities. Each row gives the input hash, map domain, median-map integral, integrated thickness quartiles and geographic validation. [Download the current results](principal-volumes.csv) · [Eruption maps and source audits](../eruptions/README.md)

The figure uses the same inputs as the table. Lower scores indicate better prediction at geographically withheld sites. The percentages are coverage of 95% predictive intervals for exact thickness measurements. A cross marks a geographic fit with a numerical warning. The earlier five-method comparisons use their own observations and folds. [Matched study editions](../archive/2026-11-24-case-editions/README.md)

![Geographic prediction scores and exact-site coverage for the methods fitted to each current input](method-scores.png)

[Reported lower limits across all distances, including the change in near-vent thickness](reported-minima/README.md)

| Question | Evidence |
|---|---|
| How do the GP ensemble and cubic spline compare inside the same observation hull? | [Regional fits and known-deposit tests](regional/README.md) |
| How do directional isopachs change from the vent to the distal deposit, and what happens when a region or campaign is withheld? | [Three-scale visual review, site agreement, volume and deletion tests](directional/multiscale/README.md) |
| How much does allowing a fan to turn change contour reach, volume and prediction across gaps? | [Directional ensemble, paired control and related studies](directional/README.md) |
| How sensitive is whole-deposit volume to spherical fan geometry and priors? | [Spherical maps, prior sensitivity and known-volume tests](spherical-mixture/README.md) |
| How do nested isopachs compare with the two other global volume estimates, and how much ash lies beyond the sampled range? | [Bulk volume, site scores, prior sensitivities and known-deposit tests](nested-isopachs/README.md) |
| Where do measurements support a contour or a partial volume? | [Local footprints and geographic deletions](support/README.md) |

The B-Tm worked comparison shows how the contours change when the three interpreted bounds are included, weakened or omitted. Earlier model-development experiments identify their own inputs. [B-Tm before and after](../eruptions/b-tm/results/cryptotephra/README.md) · [All eruption reports](../eruptions/README.md)
