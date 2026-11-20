# What has been checked?

[Documentation](README.md) · [Comparisons](../comparisons/README.md) · [Test guide](../tests/README.md)

Numerical correctness, predictive skill, uncertainty coverage and geological evidence are different checks. None alone establishes the true volume of an unobserved deposit.

| Evidence | Question it answers | Where to inspect |
|---|---|---|
| Solver and probability tests | Do likelihoods, covariance, normalization, geometry and numerical calculations match reference cases? | [Test catalogue](../tests/README.md) |
| Regional geographic holdouts | How well does the entire tuned GP/spline procedure predict withheld regions? | [Regional comparison](../comparisons/regional/README.md) and each case's held-out rows |
| Matched geographic prediction | How do both regional and both global methods predict withheld locations? | [Four-method comparison](../comparisons/README.md) |
| Principal spherical mixture | How do finite measured bearings affect false extent, missed fans and global volume? | [Six paired real fits and 24 known-volume fits](../comparisons/spherical-mixture/bearings/README.md) |
| Principal directional ensemble | How do uncertain and fixed fan directions compare on the same regions, whole quadrants and simulated sampling gaps? | [Six paired fits and 14 simulated deposits](../comparisons/directional/README.md) |
| Earlier fixed-direction ensemble | How do the weighted structures perform against equal weights and a single structure, and how often do simulated volume intervals cover truth? | [Six eruptions and 14 known-deposit trials](../comparisons/directional/fixed-ensemble/README.md) |
| Single-structure directional sensitivities | What happens when a whole direction, proximal cap or hidden fan is unobserved? | [28 trials, independent replications and limitations](../comparisons/directional/single-structure/README.md) |
| Regional known-truth trials | How do prediction, integral error and conditional coverage behave under declared synthetic scenarios? | [Regional trial record](../comparisons/regional/synthetic/README.md) |
| Global and source-focused stress tests | What happens with hidden mass, shape violations and prior dependence? | [Displaced-fan trials and failed fits](../comparisons/spherical-mixture/stress-tests.md) · [Source-focused trials](../comparisons/source-revision/stress-tests.md) |
| Support and fitting-domain studies | Which contours are rejected; how sensitive are local footprints and fits? | [Support trials](../comparisons/support/stress-tests/README.md) · [Fitting-domain sensitivity](../comparisons/support/fitting-domain/README.md) |
| Source audits and targeted alternatives | Which inputs are supported, corrected or disputed, and how do explicit scenarios change inference? | [Data audits](../data/audits/README.md) · [Aso-4](../comparisons/aso-4-audit/README.md) · [B-Tm occurrence bounds, omission and weaker-response tests](../eruptions/b-tm/results/cryptotephra/README.md) |

Read execution dates, input hashes, model geometry, priors and status. The original 40 regional trial attempts remain a development record; revising an eruption's source data does not turn those trials into new independent validation. Failed attempts and undercoverage must remain visible.

`python -m pytest -q` and `python scripts/check_repository_navigation.py` check the current code and navigation. `python -m benchmarks.verify_aso_evidence` verifies the published Aso-4 data, matched folds, posterior checksums and retained reference without rewriting its report. `python -m benchmarks.verify_directional` checks six published cases, 78 policy fits, six independent replications and 28 known-volume trials. `python -m benchmarks.verify_azimuth_continuation` replays the principal fields, volumes, held-out coverage and archive assumptions. `python -m benchmarks.verify_directional_ensemble` checks the earlier fixed-direction calibration and its 14 simulations. [Reproduction commands](../benchmarks/README.md).

Historical test counts and scientific statements describe the runs that produced them. [Older studies in Git history](../CHANGELOG.md#earlier-results) are not the default evidence for current method performance.
