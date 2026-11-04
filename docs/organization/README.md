# Repository organization verification

[Repository map](../REPOSITORY_MAP.md) · [Project home](../../README.md)

The 0.8.1 reorganization starts from commit `f1a2962d83d7db98dd4ca0628ddd2b374596da5e`. It changes navigation, source ownership and method dispatch; it does not change the numerical model definitions or replace the published analyses with new estimates.

## Preservation

- [path-map.json](path-map.json) records 658 moved files and their original SHA-256 hashes. All 540 non-Markdown artifacts are byte-identical after the move, including numerical records, maps, tables and the synthetic benchmark's execution snapshot. Markdown links were adjusted for their new locations.
- All 33 non-Markdown files in `data/` are byte-identical, including accepted inputs, source workbooks, audits and dataset profiles. Their hashes are listed in [verification.json](verification.json).
- All 36 moved numerical/data/scoring function and class bodies match the originals. The additional catalogue-directory function changes only its parent-directory depth after moving under `core/`.
- Historical records retain their execution paths and code hashes. Those are provenance, not links to be rewritten. Old README paths retain short navigation pointers.

## Regression checks

All-data fitting was captured before and after the source move for every accepted eruption and both methods: **12 complete fits**. Model descriptions, every tuning candidate/selection, query coordinates, predicted log-thickness means and variances, and regional integral/refinement records compare exactly. The two capture files have the same SHA-256; their digest and per-fit volumes are in the verification record.

A complete new Minoan comparison also passed through nested holdouts, final fitting, spline diagnostics, correlated volume draws, candidate export and figure rendering. Its input, region, fit bounds, method results, numerical validation and reliability notes exactly match the published report. Only new registry metadata and current source provenance differ. This check ran outside the published result folders; the published artifacts were preserved.

The local suite passed **87 tests**. A test-only third model exercised registry dispatch, common folds, volumes, tables, figures and benchmark summaries; it is not an active scientific method. The 0.8.1 wheel was installed in a separate environment and checked outside the checkout for CLI routes, compatibility imports and all six bundled datasets. Generated two- and three-method figures were visually inspected.

Run `python scripts/check_repository_navigation.py` for repository-local Markdown links. Source/solver checks and these regression runs establish preservation through organization; they do not add independent scientific validation beyond the [regional evidence](../../comparisons/regional/README.md).
