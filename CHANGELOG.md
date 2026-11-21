# Release history

The [eruption catalogue](eruptions/README.md) contains the current reports. For an earlier calculation, open the linked commit to obtain its input, fitting code and evidence together.

## Current release

The 12.8 ka Satsuma eruption adds 50 measured thicknesses and seven bounds. Its audit checks written field values and a lenticular marine bed 163 km west of Sakurajima. Both global families were refitted after omitting that bed. [Satsuma results](eruptions/sakurajima-satsuma/README.md)

Ilopango’s Tierra Blanca Joven Unit G adds a co-ignimbrite deposit with 52 measured thicknesses and four reported limits. The audit preserves the original numerical sources and shows how the directional coverage changes with distance from the caldera. [Ilopango results](eruptions/ilopango-tbj/README.md) · [Thickness audit](data/audits/Ilopango-TBJ/README.md)

The principal spherical-mixture result now limits the directional reach of measured sites and integrates over uncertain fan continuation in the original six-eruption study. The paired study gives the changes in 10 cm isopachs, bulk volumes, geographic predictions and errors on deposits with known volumes. [Current spherical comparison](comparisons/spherical-mixture/README.md)

The principal directional estimate allows fan direction to vary across unsampled sectors. B-Tm now fits 71 observations, including three interpreted upper limits, across all four families. The site-review protocol documents how to derive a bound, justify omission and test an influential interpretation. [Directional comparisons](comparisons/directional/README.md) · [B-Tm before and after](eruptions/b-tm/results/cryptotephra/README.md) · [Site-review protocol](docs/methods/sites-without-measured-thickness.md)

## Earlier results

| Question | Immutable release |
|---|---|
| How much did the broad spherical continuation extend into poorly sampled directions? | [Previous spherical maps, volumes and checksums](archive/2026-11-20-spherical-continuation/README.md) |
| How did the B-Tm core profiles change the geometry and volume? | [Previous 68-site B-Tm reconstruction](archive/2026-11-19-b-tm-thickness-only/README.md) · [71-site comparison with the source profiles](eruptions/b-tm/results/cryptotephra/README.md) |
| Which map preceded the primary uncertain-fan-direction estimate? | [Fixed-direction case pages alongside the first six uncertain-direction fits](https://github.com/siyulai21/tephra-fallout-isopachs/tree/efe60416128ca9587bef796f3ab5bb91a50c75ca) |
| Where are the original four-panel figures and the pages before the direction study? | [Audited fixed-direction release](https://github.com/siyulai21/tephra-fallout-isopachs/tree/ebf0fc0f27db37a70f60d0d7708a5b64f9d95516) |
| Where are the dated report bundles, figure checksums and original release inventory? | [First primary uncertain-direction release](https://github.com/siyulai21/tephra-fallout-isopachs/tree/417f624ed6c41160946cbf1fcafcf289548e58f9/comparisons/azimuth-continuation/archive) |
| What changed when B-Tm was audited? | [84-record reconstruction](https://github.com/siyulai21/tephra-fallout-isopachs/tree/2a52e062b7db84699d5229a25b01282c10301b3e/eruptions/b-tm) · [First complete audited release, with its regional prediction grid](https://github.com/siyulai21/tephra-fallout-isopachs/tree/207bb5a304718fe706953c9ad7c8d2771aefe21c/eruptions/b-tm) |
| Which directional fit preceded the six-structure ensemble? | [Single-structure release](https://github.com/siyulai21/tephra-fallout-isopachs/tree/53339910581fa8b60ce021c02c1e109b86f37a97) |
| Where are the older map versions and their checksums? | [Cartographic release records](https://github.com/siyulai21/tephra-fallout-isopachs/tree/207bb5a304718fe706953c9ad7c8d2771aefe21c/docs/maintenance) |
| How can I reproduce the former GP-only `fit` command, original spline or fixed-smoothing prototype? | [Last checkout with the old commands and solvers](https://github.com/siyulai21/tephra-fallout-isopachs/blob/2e45ecbc898139d2b93904667b96a8770245c74d/benchmarks/README.md) · [Published early reports and comparisons](https://github.com/siyulai21/tephra-fallout-isopachs/tree/41b8f2f28ac8b1c877810b782005ff243e101c63/archive) |

The early two-site B-Tm experiment used the 84-record input. The subsequent audit checked its original rows, and the current study tests three core interpretations across all four families. The original six-case summary tables and that two-site experiment can be inspected at the revision preceding this cleanup. [Previous comparison tables](https://github.com/siyulai21/tephra-fallout-isopachs/tree/f155753dad2dd4d06e4fa48d3d877242ae8539aa/comparisons) · [Early B-Tm sensitivity](https://github.com/siyulai21/tephra-fallout-isopachs/tree/f155753dad2dd4d06e4fa48d3d877242ae8539aa/comparisons/source-revision/btm-input-sensitivity)

The full version-by-version changelog records the original spline implementation, GP development, the introduction of common support checks and the later global families. [Earlier version notes](https://github.com/siyulai21/tephra-fallout-isopachs/tree/f155753dad2dd4d06e4fa48d3d877242ae8539aa/CHANGELOG.md)
