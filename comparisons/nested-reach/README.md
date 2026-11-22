# How far can a measured isopach be continued?

The following experiment documents the original finite-corridor and whole-area comparison. Current estimates add a check on transverse spread and turning ridges. [Current nested estimates](../nested-isopachs/README.md) · [Paired test of the new continuation](../global-extent/README.md)

A distant core establishes fallout at its location. Extending the same thickness across an unsampled flank can contribute much more bulk volume than the core itself constrains. The paired fits below use identical accepted observations and the same nested thickness profile. The principal corridor model allows finite angular and down-range reach around positive sites and retains a broad alternative. Bounds constrain thickness through their observation likelihood. [Geographic reach model and assumptions](../../docs/methods/nested-reach.md)

Real surveys provide the primary comparison. On Calbuco 2015 the corridor fit improved predictions on a separate field campaign and reduced the average extra span against an independently mapped 0.1 mm isopach. The Aso-4 and Pinatubo marine transfers favored the broader fit. Synthetic deposits have a separate role in testing failure modes and numerical recovery. [Independent surveys, archive bounds and field-map checks](field-surveys/README.md)

**B-Tm has the clearest evidence for the narrower map.** Mazama improves on buffered geographic predictions and changes little on whole-quadrant predictions. Aso-4’s western footprint shrinks substantially, while independent compass predictions favor broader continuation. The paired results are published together. Global volume remains conditional on how a fan continues across unvisited ground.

| Eruption | Corridor median-map integral [conditional 95%], km³ | Whole-area median-map integral (km³) | Buffered score, whole-area → corridor | Compass score, whole-area → corridor |
|---|---:|---:|---:|---:|
| [Aso-4](../../eruptions/aso-4/results/nested-isopachs/README.md) | **544.5** [249.5–9,990.1] | 2,006.6 | 1.540 → 1.556 | 1.263 → 1.579 |
| [B-Tm](../../eruptions/b-tm/results/nested-isopachs/README.md) | **33.7** [26.6–50.2] | 67.9 | 1.570 → 1.544 | 3.070 → 0.946 |
| [Ilopango TBJ (Unit G)](../../eruptions/ilopango-tbj/results/nested-isopachs/README.md) | **39.1** [25.1–94.2] | 47.7 | 1.159 → 1.274 | 1.132 → 1.283 |
| [Mazama](../../eruptions/mazama/results/nested-isopachs/README.md) | **244.6** [129.2–938.0] | 354.7 | 1.503 → 1.365 | 2.794 → 2.821 |
| [Minoan](../../eruptions/minoan/results/nested-isopachs/README.md) | **24.9** [17.4–67.6] | 31.4 | 1.536 → 1.621 | 1.494 → 1.646 |
| [Pinatubo 1991 Layer C (separate intake)](../nested-isopachs/pinatubo-preview/nested-isopachs/README.md) | **2.1** [1.5–6.9] | 5.8 | 1.076 → 1.440 | 0.958 → 1.328 |
| [Pinatubo 1991 (Layer C)](../../eruptions/pinatubo-1991-layer-c/results/nested-isopachs/README.md) | **8.4** [7.4–9.6] | 11.4 | 2.157 → 3.142 | 1.242 → 2.034 |
| [Sakurajima Satsuma (P14)](../../eruptions/sakurajima-satsuma/results/nested-isopachs/README.md) | **6.3** [3.1–30.7] | 11.5 | 0.742 → 1.016 | 1.141 → 1.353 |
| [Tambora](../../eruptions/tambora/results/nested-isopachs/README.md) | **94.9** [62.5–237.7] | 123.4 | 1.368 → 1.646 | 1.509 → 1.816 |
| [YTT](../../eruptions/ytt/results/nested-isopachs/README.md) | **1,245.3** [771.0–3,000.8] | 1,418.8 | 1.507 → 1.429 | 1.865 → 2.225 |

Each volume integrates the pointwise median-thickness map. The 95% interval uses the two tails of the volume distribution. These are bulk deposit volumes. The two scores average withheld log predictive densities with equal weight for each nonempty region. Smaller values indicate better predictions. A quadrant with no observations supplies no predictive test. [Numerical checks per case](cases.csv) · [Every held compass sector](compass-holdouts.csv) · [Contour areas by bearing](contour-areas.csv)

Across the 9 catalogued eruptions the mean buffered score changes from 1.454 to 1.621. The mean whole-quadrant score changes from 1.723 to 1.745. A complete contour band passes the shared support screen in 0 of 9 registered cases. The current Pinatubo observations enter those averages once. The earlier 137-observation development snapshot is shown separately.

## What the three disputed maps show

- **B-Tm, W.** The median 1 cm contour encloses 6,593 km² in this sector under the old fit and 71 km² under the new fit. The accepted table has 0 exact and 1 bounded readings there. [Both maps and sector volumes](../../eruptions/b-tm/results/nested-isopachs/README.md)
- **Aso-4, W.** The median 1 cm contour encloses 6,319,337 km² in this sector under the old fit and 48,254 km² under the new fit. The accepted table has 0 exact and 0 bounded readings there. [Both maps and sector volumes](../../eruptions/aso-4/results/nested-isopachs/README.md)
- **Mazama, SW.** The median 1 cm contour encloses 28,185 km² in this sector under the old fit and 5,771 km² under the new fit. The accepted table has 0 exact and 0 bounded readings there. [Both maps and sector volumes](../../eruptions/mazama/results/nested-isopachs/README.md)

## Recovery of deposits with known thickness

The whole-area and corridor fits covered the known total in 18/20 and 17/20 cases, respectively. The 20 truths come from 18 surveys, including two pairs with identical observations and different hidden deposits. These are development stress tests.

For a genuinely broad, circular deposit surveyed only on its eastern half, the two corridor fits missed 159,977–304,290 km² of the true western 1 cm footprint. Both conditional volume intervals included the known total. A volume interval can cover the truth while the median contour misses a real lobe. Both models also underestimated the deliberately diffuse global tail. The inherited analogue thinning prior limits both methods there.

The earlier regional GP and cubic-spline benchmarks were rerun on matched inputs. Both the registered-case summary and the 80-row synthetic table reproduced their previous SHA-256 hashes. [Known volume and false/missed area](known-surfaces.csv) · [Per-site scores, model diagnostics and known-surface receipts](validation.json.gz) · [Regional replay](regional-parity.json)
