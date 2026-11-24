# How well do the methods predict fallout?

The principal table lists results fitted to the accepted edition shared across all five methods for each eruption. Each method reports the integral of its median map and the integrals of its pointwise Q25 and Q75 thickness maps. Scores and 95% observation coverage describe prediction in withheld regions. [Volumes, validation and source hashes](principal-volumes.csv)

The nested family fits its footprint and area–thickness law jointly. Matched comparisons with the original whole-area and station-corridor models include real withheld surveys, known deposits and measurements of the actual unsmoothed contours. [Catalogue estimates](nested-isopachs/README.md) · [Map comparisons and validation](nested-isopachs/geography/README.md)

The Ilopango comparison uses the previous 56-site edition. The November 24 field audit and standalone nested fit use 116 sites. [Current Ilopango result](../eruptions/ilopango-tbj/results/nested-isopachs/2026-11-24/README.md)

The November 24 [standalone Pinatubo nested fit](../eruptions/pinatubo-1991-layer-c/results/nested-isopachs/2026-11-24/README.md) uses 235 sites. The matched five-method Pinatubo comparison uses 208 observations, including 34 offshore records. Its land measurements incorporate both the Koyaguchi and Paladio-Melosantos figure reviews. The earlier field-survey experiments keep their own fitted inputs and held-out predictions. [Audited Pinatubo case](../eruptions/pinatubo-1991-layer-c/README.md) · [Effect of the two land-map reviews](../eruptions/pinatubo-1991-layer-c/results/source-checks/figure7-review/README.md) · [Earlier survey tests](nested-isopachs/pinatubo-preview/README.md)

The figure compares geographic predictions on the matched five-method inputs. All five Mazama fits use the same 300 localities and the same buffered geographic holdouts. B-Tm includes two interpreted lake bounds and one peat bound in every family. Each region receives equal weight. Lower scores mean better prediction. Percentages give the share of exact thicknesses inside a 95% predictive interval. A cross marks incomplete scoring or a numerical warning.

![Geographic prediction scores and exact-site coverage for the five methods](method-scores.png)

| Question | Evidence |
|---|---|
| How do the GP ensemble and cubic spline compare inside the same observation hull? | [Regional fits and known-deposit tests](regional/README.md) |
| How much does allowing a fan to turn change contour reach, volume and prediction across gaps? | [Directional ensemble, paired control and related studies](directional/README.md) |
| How sensitive is whole-deposit volume to spherical fan geometry and priors? | [Spherical maps, prior sensitivity and known-volume tests](spherical-mixture/README.md) |
| How do nested isopachs compare with the two other global volume estimates, and how much ash lies beyond the sampled range? | [Bulk volume, site scores, prior sensitivities and known-deposit tests](nested-isopachs/README.md) |
| Where do measurements support a contour or a partial volume? | [Local footprints and geographic deletions](support/README.md) |

The B-Tm worked comparison shows how the contours change when the three interpreted bounds are included, weakened or omitted. Earlier model-development experiments identify their own inputs. [B-Tm before and after](../eruptions/b-tm/results/cryptotephra/README.md) · [All eruption reports](../eruptions/README.md)
