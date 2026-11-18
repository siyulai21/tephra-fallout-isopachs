# History without duplicate working copies

The directional median-isopach ensemble and its published comparison were added on 18 November 2026. The prior single-structure reports and posterior draws remain available from each directional case. [Ensemble comparison](../comparisons/directional-ensemble/README.md) · [Dated presentation archive](../comparisons/directional-ensemble/history/2026-11-18/manifest.json)

[Repository map](README.md#repository-layout) · [Maintenance policy](maintenance/README.md)

Unused archived report collections, old navigation redirects and the superseded organization manifest were removed from the working tree on 2026-11-08. This does not rewrite Git history or delete current scientific evidence.

[Browse the complete pre-cleanup repository](https://github.com/siyulai21/tephra-fallout-isopachs/tree/41b8f2f28ac8b1c877810b782005ff243e101c63) at commit `41b8f2f28ac8b1c877810b782005ff243e101c63`. Its [archive](https://github.com/siyulai21/tephra-fallout-isopachs/tree/41b8f2f28ac8b1c877810b782005ff243e101c63/archive) contains earlier selected-GP, rectangular-domain, original-spline and superseded global calculations. The [change log](../CHANGELOG.md) records the development sequence. Use a historical commit when reproducing an old execution; current code and inputs may differ.

Two dependencies remain local:

- [Pre-audit Aso-4 reference](../comparisons/aso-4-audit/reference/README.md): its posterior supports the current high-tail/influence diagnosis and its original artifacts are checksum-verified. It is not the accepted current input or headline result.
- [YTT regression fixture](../tests/fixtures/README.md): the exact model parameters needed by the old-fit regression, with the original report's commit, path and hash.

Source/audit history in `data/`, current development trials, failed numerical attempts, qualified alternatives and source-code snapshots remain in the repository because they are still evidence. Older execution paths inside immutable JSON are historical provenance, not broken instructions to substitute current data.


## Method evolution and the 2026-11-17 organization

| Stage | Relation to today's workflow |
|---|---|
| Original cubic spline and first trend-penalty experiment | Numerical engines retained for historical benchmarks/regression; neither produces current case reports |
| Selected GP and GP stacking | A single selected GP is an internal/historical option; the ensemble is the current regional GP family |
| Automatic trend-regularized cubic spline | Current regional peer of the GP, with training-fold selection and explicit bounds |
| Spherical mixture | Adds a separate conditional whole-Earth mass posterior; it does not replace regional reconstruction |
| Directional continuation (0.12) | Adds another global family using local triangles and directional thinning; it does not consume or supersede the other fitted fields |
| Directional ensemble (0.14) | Six structural fits supply the median-isopach map. Whole-field volume intervals and the earlier single-structure sensitivities are reported separately. |

[PR #16](https://github.com/siyulai21/tephra-fallout-isopachs/pull/16) introduced the directional family. [The exact pre-organization snapshot](https://github.com/siyulai21/tephra-fallout-isopachs/tree/d6b460bc2379dd3e86b86b1ee05fd9b65aeaa8a4) preserves its original paths.

The 2026-11-17 cleanup names spherical outputs `eruptions/<slug>/results/spherical-mixture/` and the study `comparisons/spherical-mixture/`. The shared index introduced at that time was retired when the full ensemble reports moved to their family home. All original scientific files, posterior samples, figures and execution hashes are retained. Historical manifests resolve their old paths in the verifier instead of changing the evidence.

The unused original desktop GUI and its optional Basemap extra were removed. The duplicate `docs/METHODS.md` overview was consolidated into the [four-family catalogue](../methods/README.md). Original spline numerical engines, compatibility helpers and full unsuccessful trials remain because current reproduction or verification still uses them. [Organization verification](maintenance/cleanup-2026-11-17.json).
