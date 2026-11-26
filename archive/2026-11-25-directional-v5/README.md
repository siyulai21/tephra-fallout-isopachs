# Directional v5 numerical sources

These files preserve the directional numerical engine from immutable revision
`beee82b7020d62aa2c3dc847d74e45dcf651084f`, before the investigation of geographic
regularity. They are retained for the engine identity and replay of the
[original multiscale study](../../comparisons/directional/multiscale/README.md).

The current accepted-input v5 results at that revision include the
[reported-minimum policy](../../docs/methods/reported-minimum-fitting.md).
The earlier multiscale study uses its explicitly archived observation reader.
These are distinct evidence editions; comparisons must use matching editions.

No source observation, reported bound or eruption-specific parameter is changed
by preserving this engine. These sources are an archive, not the principal
implementation.

`current-cases/<slug>/` retains the nine immediately preceding directional
posteriors, median grids, maps, probabilities and complete-draw reports on the
current accepted inputs. Its [provenance](current-cases/provenance.json) records the original
paths and hashes. These packets supply the matched controls for the
[v6 geography study](../../comparisons/directional/geographic-regularity/README.md).

The [preceding control table](preceding-control-table.csv) preserves the root
comparison table at the same revision. The current uncertain-azimuth table now
reads only its own paired reports and declares each report's input edition,
instead of substituting later finite-spread or multiscale map receipts.

`preceding-spherical-summary/` retains the preceding paired spherical figure and
table. Their current replacements enumerate the eight locally published
spherical packets; B-Tm's separate external spherical work remains linked from
its case page and is not represented as a completed local posterior.
