# Working on tephra-fallout-isopachs

The intended use is assistant-led: the user uploads eruption data in chat; the assistant audits sources, runs all active empirical families, and publishes data and readable reports on GitHub. See [the intake workflow](docs/NEW_ERUPTION.md), [contribution guide](CONTRIBUTING.md) and [method contract](docs/ADDING_METHOD.md).

- Preserve original sources and hashes. Keep units, source-row identities, bounds and inclusion/exclusion decisions auditable. Do not invent geological evidence, pseudo-observations, zero deposits or uncertainty values.
- Use `data/datasets.json` for accepted cases and `src/empirical_isopach/models/registry.py` for active methods. Current families are the GP ensemble and automatic cubic spline, with no preferred family.
- Compare methods on identical accepted observations, outer geographic folds and the same regional hull. Tune only within training folds. Keep each method's volume and uncertainty separate; global extrapolation does not decide regional performance.
- Work in a fresh ignored `results/` destination. Publish reviewed case results under `eruptions/<slug>/results/`, link data/audit/history from its home, and update the relevant indexes. Preserve replaced results in dated archive folders.
- Retain failures, search boundaries, poor coverage and conditional-uncertainty qualifications. Verify numerical checks and maps; a smooth-looking surface is not validation.
- Run the shared support audit for every active family. Do not turn geometric closure, a complete-Earth integral, or an observation hull into a claim of supported global volume. Preserve unsupported holes, all contour rejection reasons, and separate proximal/distal assumptions. See [the support protocol](docs/methods/supported-isopachs.md).
- Keep source code organized into the shared framework and separate model families. Historical import paths are compatibility wrappers; change active implementations at their new paths.
- Run checks appropriate to the change: `python -m pytest -q` and `python scripts/check_repository_navigation.py`. Numerical method changes also need the matched regional benchmark. For organization-only changes, verify data/artifact hashes and numerical parity instead of rewriting published results.
- Do not add an atlas or browser workflow. Reports, maps, tables and provenance must remain readable directly on GitHub. Leave unrelated working copies and work in progress untouched.
