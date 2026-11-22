# Working on tephra-fallout-isopachs

Users submit eruption data in chat. Follow the [intake workflow](docs/NEW_ERUPTION.md) for new and recalculated eruptions, run all active families and publish a readable case on GitHub.

## Sources of truth

- `data/datasets.json` identifies the accepted CSV, audit and reviewed spherical-mixture geometry. Aso-4 uses its reviewed input. B-Tm uses source-focused spherical fans. Preserve original units, row identities, hashes, bounds and reasons for inclusion or exclusion. Never invent evidence, observation errors, pseudo-data or zero deposits.
- Before registering a new input or revising an old one, inventory actual sites with missing, trace or proxy thicknesses, including earlier exclusions. Read the primary site descriptions. Investigate sites that could constrain a fan edge, gap or distal reach first. Derive and include defensible site-specific bounds once in the principal data for all five families. Omit a site from that fit only after documenting why the available evidence cannot bound original local fallout, or why the relevant isopachs depend little on that site. Preserve its occurrence and any component constraints. Test influential interpretations against omission and weaker transfer. [Site decision protocol](docs/methods/sites-without-measured-thickness.md) · [Audit worksheet](templates/SITE_EVIDENCE.md)
- `src/empirical_isopach/models/registry.py` lists the regional GP ensemble and cubic spline. Compare them on the same accepted rows, outer geographic folds and observation hull. Tune inside training folds. [Method contract](docs/ADDING_METHOD.md)
- The global families fit the observations independently. Publish the complete directional narrative once at `results/directional/README.md`. Publish each global family’s median-thickness surface in `deposit.png`, and report its whole-Earth integral from `deposit.json` at `median_map_volume_km3`. The direct posterior quadrature and complete-deposit intervals remain in `azimuth/report.json`. Use quantitative thicknesses and bounds for this estimate. Publish archive-conditioned scenarios, the matched fixed-direction control, the older calibration in `ensemble/`, and the reference in `single-structure/` as separate sensitivities. Retain both geographic and quadrant tests and all numerical warnings. [Principal directional protocol](docs/methods/uncertain-azimuth.md)
- `results/spherical-mixture/report.json` contains the principal bearing-conditioned spherical posterior. Lead with the median-map integral from `spherical-mixture/deposit.json`. Use integrated posterior draws for the conditional 95% interval endpoints. Retain the broad parent in `bearing/parent-report.json`, the enhanced hidden-fan alternative, both reach sensitivities, the shared support screen, all fixed prior fits and training-only geographic tests. [Bearing protocol](docs/methods/spherical-bearings.md) Preserve separate conditional intervals, prior-sensitivity envelopes and numerical status. [Spherical-mixture protocol](docs/methods/probabilistic-global.md)
- Apply the shared support audit to every family. Keep unsupported holes, rejection reasons and assumptions about proximal and distal fallout. Publish conditional global estimates even when no complete band qualifies. [Support protocol](docs/methods/supported-isopachs.md)
- For a broad unsampled direction, publish the measured-fan continuation and its sector volumes under `directional/bearing/`. Retain disagreement with the missing-fan sensitivity. An inspected lake or peat horizon can test a predicted isopach. Record its correlation, sampling, preservation and recognition controls, score it at the actual site, and qualify any bound on primary fallout. Unsurveyed directions supply no absence observation. [Archive evidence protocol](docs/methods/archive-evidence.md)

Use solid directional contour segments only for documented measured brackets. Show conditional continuation with dashes and include the uncertain-azimuth median and exceedance maps for each case. [Directional extent protocol](docs/methods/uncertain-azimuth.md)

## Run and publish

Use a fresh ignored destination. The normal command runs both regional methods, local support, the directional ensemble and its reference sensitivities, the spherical mixture, and Bayesian nested isopachs.

```sh
tephra-isopachs compare ERUPTION_OR_CSV --output results/new-case
```

Publish reviewed outputs under `eruptions/<slug>/results/`. Use **one case home** for the accepted inputs, the three global deposit plates, the two regional isopachs and the matched table of map integrals and complete-draw volumes, the support decision and material source or numerical qualifications. Use the shared comparison section and preserve the case-specific interpretation. Link directly to the regional, directional ensemble, spherical-mixture, nested-isopach and support reports. Add one row to `eruptions/README.md`. Refresh each cross-eruption summary from the registered case reports when replacing an input. Publish the current regional table at `comparisons/regional/comparison.csv`. Keep the matched benchmark used by a dated case study with that study. Group directional validation and sensitivity studies under `comparisons/directional/`. Publish the spherical estimates, paired maps and known-volume tests together in `comparisons/spherical-mixture/README.md` using the [summary commands](benchmarks/README.md#current-evaluations). Link each family once from the comparison index.

Preserve earlier result identities, raw evidence, relevant sensitivity studies, failed attempts and hashes when revising a case. Record the preceding immutable revision and retain any locally required historical evidence. Leave unrelated work in progress untouched. [Publication and retention policy](docs/maintenance/README.md)

## Verify

```sh
python -m pytest -q
python scripts/check_repository_navigation.py
python -m benchmarks.verify_spherical_bearings
python -m benchmarks.verify_visuals
python -m benchmarks.verify_btm_cryptotephra
python -m benchmarks.verify_btm_audit
python -m benchmarks.verify_bearing_continuation
python -m benchmarks.verify_aso_evidence
python -m benchmarks.verify_directional
python -m benchmarks.verify_directional_ensemble
python -m benchmarks.verify_azimuth_continuation
```

For a navigation change, verify scientific artifacts against their pre-change bytes. For a numerical change, run the matched regional benchmark and the relevant global or support studies. New map releases need visual inspection and replay checks. [Study commands](benchmarks/README.md) · [Figure checks](docs/maintenance/README.md#checking-a-map)

For Bayesian nested isopachs, publish dispersal corridors as the principal in `nested-isopachs/` and the whole-area sensitivity in `nested-whole-area/`. Retain the independent population checks, fixed prior sensitivities, coherent volume and thickness budgets, all geographic fits and the shared contour screen. Run `python -m benchmarks.verify_nested_reach` and `python -m benchmarks.verify_nested` on published packets. [Protocol](docs/methods/nested-isopachs.md)
