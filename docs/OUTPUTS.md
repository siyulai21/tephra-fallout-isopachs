# Reading current maps, volumes and output files

[Eruption cases](../eruptions/README.md) · [Method relationships](methods/README.md) · [Commands](QUICKSTART.md)

Start with the case home. It links the accepted evidence, regional comparison, support audit and all three global families. The normal `compare` command writes the layout below. A standalone `global --output PATH` command writes its chosen family's report directly to `PATH`.

The `extent/` directory in each global family contains the current conditional interval, scores from withheld regions, parent comparisons and the complete contour audit. The spherical directory also contains the eight fixed prior sensitivities. [Current continuation rules](methods/finite-spread.md)

| Location relative to a case's `results/` | Scope |
|---|---|
| `README.md`, `report.json`, maps and `selection/` | GP ensemble and cubic spline, fitted separately on a common regional hull |
| `support/` | Local footprints, holes, partial volumes and regional contour decisions |
| `directional/` | Principal uncertain-fan-direction report in `README.md`, with its quantitative field, whole-Earth volume and contour audit in `azimuth/` |
| `spherical-mixture/` | Spherical-mixture whole-Earth posterior and assumption sensitivities |
| `nested-isopachs/` | Principal corridor map, integrated volume draws, local brackets, geographic predictions and support |
| `nested-whole-area/` | Broad nested sensitivity with its own maps, volume draws and geographic scores |

Older records use `global/` for the spherical-only files. Those files were moved without altering their scientific contents; execution paths and hashes remain as originally recorded. The [cross-eruption comparison](../comparisons/README.md) links the five families on matched geographic tests.

## Read the map and its volume

Each case home compares the three full deposits and the regional pair. In each global family folder, `deposit.png` shows the full 0.5 cm footprint, a near-vent view and the thickness–area curve. Land and inland lakes come from Natural Earth. Circles are measurements, triangles are reported lower or upper limits, and diamonds mark interpreted upper limits. Symbols take the colour of the measured thickness or stated limit. A site's true thickness can lie beyond its reported bound.

For a thickness $t$, the curve plots $t$ against $\sqrt{A(T\ge t)}$ in kilometres. Areas in `thickness-area.csv` include every disjoint patch across Earth. Each row reports the coarser-grid comparison. The renderer doubles radial and angular resolution when a contour area changes by more than both 5% and 1 km², or its map integral differs from independent field quadrature by more than 1.2%. Any remaining area sensitivity is recorded in the CSV. A zero area means the corresponding sampled field never reached that level. The near-vent half-width is the lower quartile of distances to distinct exact sites, so all three methods show the same window for an eruption.

Every principal estimate is $V=10^{-5}\int_D T(x)\,dA$ in km³, with mapped thickness in cm and area in km². The regional domain is the observation hull and its integral is `regional_volume.median_surface_km3`. Each global `deposit.json` reports `median_map_volume_km3`, integrated across the physical WGS84 Earth including all thin tails. The inferred surface between saved polar nodes is bilinear in log thickness and periodic in bearing. Positive quadrature integrates that same surface with the equal-area radial Jacobian. Two independent Sobol scrambles evaluate the original fit directly. The receipt records both scrambles, mesh refinement, the display-edge test and the input and posterior hashes. Finite resolution limits the smallest detectable patches.

Complete simulated deposits have their own volume distribution. The interval column gives its conditional 95% range. A map integral can fall outside that interval. Read sampling, integration and spatial support warnings alongside each estimate. [Matched map volumes and complete-draw intervals](../comparisons/principal-volumes.csv)

## Regional GP and cubic spline

Both integrate the same observation hull. Regional maps share colour/contour scales, and their contours stop at the hull. Neither a hull nor a smooth contour establishes uniform observational support. The reported regional point estimate integrates the median surface over that hull.

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

Open `directional/README.md` for the principal estimate. In `directional/azimuth/report.json`, **`volume.median_map_km3`** integrates the quantitative pointwise median and **`volume.coherent_volume_km3`** gives the two conditional 95% volume endpoints. `volume.median_map_parts_by_sector_km3` partitions the central volume into five support classes and eight compass sectors. `fields.npz` stores the coherent draws and their volumes. `median-grids.npz`, `isopachs.png` and `world.png` reproduce the direct-field diagnostic maps. The geographic score and exact-site observation coverage in the comparison come from this same model. The overview diamond marks the median-map volume. [Uncertain-fan-direction protocol](methods/uncertain-azimuth.md)

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

The mapped integral is in `deposit.json`. The quantiles in `fits.baseline.volume` summarize complete bearing-conditioned fields. Each fit’s `training_checks.rmse_log` gives its in-sample thickness error. The raw `posterior.npz` describes parent amplitudes and kernel parameters. The baseline packet `bearing/baseline.npz` pairs them with retained volumes, orientations, sectors and support classes. `bearing/parent-report.json` stores the original broad spherical result. [Calculation and assumptions](methods/spherical-bearings.md)

| File in `spherical-mixture/` | Read it for |
|---|---|
| `README.md`, `field.png`, `volume.png`, `sampling.png` | Conditional field/volume, sensitivity, unobserved mass and chain diagnostics |
| `report.json`, `estimates.csv` | Priors, baseline and sensitivities, all numerical attempts and provenance |
| `posterior.npz` | Broad-parent amplitude and kernel draws, with chain identity. `bearing/baseline.npz` pairs these draws with the principal retained mass. |
| `volume_draws.csv.gz` | Integrated volumes by assumption and chain. Within an assumption, the parent draw index and fan-width power jointly identify a deposit. The three widths share one kernel draw and orientation. |
| `heldout_predictions.csv` | Matched geographic outer-fold predictions |
| `source.png` when present | Source-region shape and maximum checks |
| `artifacts.json` | Published file hashes |

`fits.baseline.volume` contains its two conditional 95% credible interval endpoints. `sensitivity_envelope_km3` has no joint 95% probability interpretation. `mass_partition` describes sampling geometry, not geological confidence. `diagnostics.passed` combines the parent sampling status, retained-mass integration check and split-draw volume check. `normalization_refinements` preserves preceding lower-resolution fits. `source_checks` records source-geometry diagnostics. Read small mass fractions with their refinement errors. [Spherical protocol](methods/probabilistic-global.md) · [Source-focused geometry](methods/source-anchored-fans.md).

All volumes are **bulk fallout**, not dense-rock equivalent or total erupted material including ignimbrite deposits. Regional conditional intervals, global credible intervals, geographic-deletion ranges and prior-sensitivity envelopes remain separate. Passing numerical checks does not identify an unseen deposit.

## Bayesian nested isopachs

The corridor point estimate is the integral in `nested-isopachs/deposit.json`. Its `report.json` lists the two conditional 95% endpoints in `conditional_volume_interval95_km3`, source hashes, held-out predictions, prior sensitivities and numerical checks. The whole-area sensitivity uses `nested-whole-area/median-field.json`. Spatially correlated draws allow independent checks of uncertainty and support. [Principal estimates](../comparisons/nested-isopachs/README.md) · [Corridor model](methods/nested-reach.md)
