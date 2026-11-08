# History without duplicate working copies

[Repository map](REPOSITORY_MAP.md) · [Maintenance policy](maintenance/README.md)

Unused archived report collections, old navigation redirects and the superseded organization manifest were removed from the working tree on 2026-11-08. This does not rewrite Git history or delete current scientific evidence.

[Browse the complete pre-cleanup repository](https://github.com/siyulai21/tephra-fallout-isopachs/tree/41b8f2f28ac8b1c877810b782005ff243e101c63) at commit `41b8f2f28ac8b1c877810b782005ff243e101c63`. Its [archive](https://github.com/siyulai21/tephra-fallout-isopachs/tree/41b8f2f28ac8b1c877810b782005ff243e101c63/archive) contains earlier selected-GP, rectangular-domain, original-spline and superseded global calculations. The [change log](../CHANGELOG.md) records the development sequence. Use a historical commit when reproducing an old execution; current code and inputs may differ.

Two dependencies remain local:

- [Pre-audit Aso-4 reference](../comparisons/aso-4-audit/reference/README.md): its posterior supports the current high-tail/influence diagnosis and its original artifacts are checksum-verified. It is not the accepted current input or headline result.
- [YTT regression fixture](../tests/fixtures/README.md): the exact model parameters needed by the old-fit regression, with the original report's commit, path and hash.

Source/audit history in `data/`, current development trials, failed numerical attempts, qualified alternatives and source-code snapshots remain in the repository because they are still evidence. Older execution paths inside immutable JSON are historical provenance, not broken instructions to substitute current data.
