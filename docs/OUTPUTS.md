# Reading current maps, volumes and output files

[Eruption cases](../eruptions/README.md) · [Method relationships](methods/README.md) · [Commands](QUICKSTART.md)

Start with the case home. It links the accepted evidence, regional comparison, support audit and **both** global families. The normal `compare` command writes the layout below. A standalone `global --output PATH` command writes its chosen family's report directly to `PATH`.

| Location relative to a case's `results/` | Scope |
|---|---|
| `README.md`, `report.json`, maps and `selection/` | GP ensemble and cubic spline, fitted separately on a common regional hull |
| `support/` | Local footprints, holes, partial volumes and regional contour decisions |
| `directional/` | Principal uncertain-fan-direction report in `README.md`, with its quantitative field, whole-Earth volume and contour audit in `azimuth/` |
| `spherical-mixture/` | Spherical-mixture whole-Earth posterior and assumption sensitivities |

Older records use `global/` for the spherical-only files. Those files were moved without altering their scientific contents; execution paths and hashes remain as originally recorded. The [cross-eruption comparison](../comparisons/README.md) shows all four families on matched geographic tests.

## Visual reading route

Start on the case home. Its `four-method-evidence.png` compares four median fields in the same observation frame. The directional panel uses the uncertain-fan-direction field. `directional/azimuth/isopachs.png` shows that field at deposit and near-source scales. Expand the whole-Earth pair for the uncertain-direction and spherical-mixture maps. The four-family volume and prediction plot is `overview.png`.

`map-display.json` records the rendering grid, frame, coastlines and global plotting subset. It is separate from the original scientific `report.json`. A figure refresh also includes `figure-replay.json` with original report/posterior hashes and regional integral parity. Display grids never set mass. [Rendering policy and interpretation](maintenance/README.md#checking-a-map).

The four principal fits use the same accepted measurements and bounds, including any site-specific interpreted limits. The source audit gives its derivation and the shared source-review protocol explains the decision. [Site decision protocol](methods/sites-without-measured-thickness.md) The next table labels the model and evidence for each sensitivity, including the uncertain-direction result conditioned on recognizable beds when available. `directional/bearing/README.md` gives the full set of assumptions, site probabilities, sector volumes and numerical warnings. [B-Tm omission and weaker-response comparisons](../eruptions/b-tm/results/cryptotephra/README.md)

## Regional GP and cubic spline

Both integrate the same observation hull. Regional maps share colour/contour scales, and their contours stop at the hull. Neither a hull nor a smooth contour establishes uniform observational support. The median-surface integral differs from the median of random whole-field volumes.

| File | Read it for |
|---|---|
| `README.md`, `isopachs.png/.svg`, `diagnostics.png/.svg` | Paired findings, surfaces, prediction and reliability |
| `report.json` | Method identity, region, tuning, nested validation, numerical checks and provenance |
| `accepted_observations.csv` | Accepted row identities, values, projected coordinates and bound semantics |
| `heldout_predictions.csv` | Fold membership, predictions, intervals and scores |
| `selection/*.csv` | Candidate settings, inner-fold scores and failures |
| `predictions.csv` | Regenerable grid, usually excluded from published reports |

In `report.json`, read `methods.<method>.regional_volume`, `joint_volume` and `validation.methods.<method>`. Method keys are `stacked` and `spline`. Solid bars show conditional whole-field intervals; dotted geographic-deletion ranges are not confidence intervals. Fitted-parameter and selection uncertainty are not fully propagated; trend shrinkage can yield a particularly narrow spline interval. [Regional protocol](methods/regional-comparison.md).

## Local support and contour decisions

| File | Meaning |
|---|---|
| `support/README.md`, `support.png` | Interpolation patches, holes, partial volumes and rejected contours |
| `support/contour_audit.csv` | Every candidate level, policy, stability check and rejection reason |
| `support/report.json` | Three policies, deletion sensitivity, band checks and conditional scenarios |
| `support/support_geometry.json` | Footprint polygons in source-centred projected kilometres |
| `directional/azimuth/complete-contour-audit.json` | Complete-contour screen on the principal uncertain-direction field. The fixed-direction screen is in `directional/ensemble/complete-contour-audit.json`. |
| `spherical-mixture/support/` | This case’s complete contour decisions and checksum-verified posterior fields. [All case support audits](../comparisons/support/README.md) |

The regional support report records `methods.<method>.footprint_volumes` and `contours.policies`. Its `preferred_global_km3` is null: this audit does not select a global total. A zero quadrature fraction is not proof of absent ash. Preserve rejection reasons, unsupported holes and numerical warnings. A failed complete-contour screen is diagnostic, not a veto on conditional global inference. [Support protocol](methods/supported-isopachs.md).

## Directional ensemble

Open `directional/README.md` for the principal estimate. In `directional/azimuth/report.json`, **`volume.median_map_km3`** integrates the quantitative pointwise median and **`volume.coherent_volume_km3`** summarizes whole-deposit draws. `volume.median_map_parts_by_sector_km3` partitions the central volume into five support classes and eight compass sectors. `fields.npz` stores the coherent draws and their volumes. `median-grids.npz`, `isopachs.png` and `world.png` reproduce the primary maps. The geographic score and exact-site observation coverage in the comparison come from this same model. The overview diamond marks the median-map volume. [Uncertain-fan-direction protocol](methods/uncertain-azimuth.md)

The matched fixed-direction control is `volume.matched_control_map_km3` in the same report. The original fixed-direction weights and volume are recorded separately in `directional/ensemble/report.json`. Archive-conditioned scenarios retain their own assumptions and intervals.

## Single-structure directional reference

| File in `directional/single-structure/` | Read it for |
|---|---|
| `README.md`, `isopachs.png`, `directions.png` | Supported fragments, extrapolated field, sectors, sensitivity and warnings |
| `report.json`, `estimates.csv` | Reference and 12 alternative policies, all attempts, intervals and provenance |
| `posterior.npz` | Joint reference posterior fields, with chain/draw identity |
| `volume-draws.npz` | Separate volume samples for all 13 policies |
| `directional-mass.npz` | Reference mass for five disjoint regions × eight bearings in every field |
| `source-cap.npz` | Unobserved proximal addition and cap-inclusive totals |
| `heldout.csv` | Geographic heldout observations and predictions |
| `isopach-fragments.csv` | Complete triangle-by-level display decisions, including rejected fragments |
| `artifacts.json` | Published file hashes |

`fits.reference.volume_km3` is the reference conditional interval; other `fits` entries retain their own assumptions and numerical status. `validation.geographic` and `validation.quadrants` contain distinct deletion tests. `unobserved_proximal_cap` describes a positive addition that leaves every accepted observation unchanged. The reference mass array partitions the whole Earth; its footprint is not the same region as the GP/spline hull. Sum sectors **within each draw before taking quantiles**: marginal medians and interval endpoints do not add. [Directional definition](methods/directional-continuation.md).

## Spherical-mixture global posterior

| File in `spherical-mixture/` | Read it for |
|---|---|
| `README.md`, `field.png`, `volume.png`, `sampling.png` | Conditional field/volume, sensitivity, unobserved mass and chain diagnostics |
| `report.json`, `estimates.csv` | Priors, baseline and sensitivities, all numerical attempts and provenance |
| `posterior.npz` | Baseline whole-field parameter samples, retaining chain identity |
| `volume_draws.csv.gz` | Separate volume samples for every assumption and chain |
| `heldout_predictions.csv` | Matched geographic outer-fold predictions |
| `source.png` when present | Source-region shape and maximum checks |
| `artifacts.json` | Published file hashes |

`fits.baseline.volume` contains its conditional median and 95% credible interval. `sensitivity_envelope_km3` has no joint 95% probability interpretation. `mass_partition` describes sampling geometry, not geological confidence. `diagnostics.passed` concerns mixing and, where applicable, normalization accuracy. `normalization_refinements` preserves preceding lower-resolution fits. `source_checks` records source-geometry diagnostics. Read small mass fractions with their refinement errors. [Spherical protocol](methods/probabilistic-global.md) · [Source-focused geometry](methods/source-anchored-fans.md).

All volumes are **bulk fallout**, not dense-rock equivalent or total erupted material including ignimbrite deposits. Regional conditional intervals, global credible intervals, geographic-deletion ranges and prior-sensitivity envelopes remain separate. Passing numerical checks does not identify an unseen deposit.

## Historical GP workflow

Historical GP output conventions are preserved in the [version-specific guide](https://github.com/siyulai21/tephra-fallout-isopachs/blob/41b8f2f28ac8b1c877810b782005ff243e101c63/archive/reference/historical-gp-outputs.md).
