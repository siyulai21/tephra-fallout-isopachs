# Spherical mixture before bearing-dependent continuation

The preceding release extended each fitted kernel broadly across unsampled directions. Its original maps, volumes, source code and case pages can be read at the exact Git revision used in the paired comparison. The dated SHA-256 inventory identifies all 115 replaced files. [Browse the preceding release](https://github.com/siyulai21/tephra-fallout-isopachs/tree/843a2eef0fc669e38d95e4153ac8d7464f4d4a98) · [Original checksums](previous-files.sha256) · [Current maps and comparison](../../comparisons/spherical-mixture/README.md)

The six original spherical reports now live beside their current fits at `eruptions/<slug>/results/spherical-mixture/bearing/parent-report.json`. The B-Tm release audit, the Aso-4 neighboring-eruption checks and the paired maps use 62 historical files in total. Their other inputs are kept here for offline replay. [Figures, tables and display frames by eruption](audit-inputs/) · [Original pages referenced by the source audits](audit-pages.zip)

Run `python -m benchmarks.verify_spherical_bearings` to check the published reconstructions and those 62 inputs. Add `--check-history` to verify every file in the inventory against the preceding commit. In a shallow checkout, first run `git fetch --no-tags --depth=1 origin 843a2eef0fc669e38d95e4153ac8d7464f4d4a98`.
