# What has been checked?

[Documentation](README.md) · [Comparisons](../comparisons/README.md) · [Test guide](../tests/README.md)

Numerical correctness, predictive skill, uncertainty coverage and geological evidence are different checks. None alone establishes the true volume of an unobserved deposit.

| Evidence | Question it answers | Where to inspect |
|---|---|---|
| Solver and probability tests | Do likelihoods, covariance, normalization, geometry and numerical calculations match reference cases? | [Test catalogue](../tests/README.md) |
| Regional geographic holdouts | How well does the entire tuned GP/spline procedure predict withheld regions? | [Regional comparison](../comparisons/regional/README.md) and each case's held-out rows |
| Global geographic holdouts | How well does the global model predict the same withheld evidence under each reviewed geometry? | [Global reports](../comparisons/global/README.md) |
| Regional known-truth trials | How do prediction, integral error and conditional coverage behave under declared synthetic scenarios? | [Regional trial record](../comparisons/regional/synthetic/README.md) |
| Global and source-focused stress tests | What happens with hidden mass, shape violations and prior dependence? | [Global development record](../comparisons/global/development/README.md) · [Source-focused trials](../comparisons/source-revision/stress-tests.md) |
| Support and fitting-domain studies | Which contours are rejected; how sensitive are local footprints and fits? | [Support trials](../comparisons/support/stress-tests/README.md) · [Fitting-domain sensitivity](../comparisons/support/fitting-domain/README.md) |
| Source audits and targeted alternatives | Which inputs are supported, corrected or disputed, and how do explicit scenarios change inference? | [Data audits](../data/audits/README.md) · [Aso-4](../comparisons/aso-4-audit/README.md) · [B-Tm/source geometry](../comparisons/source-revision/README.md) |

Read execution dates, input hashes, model geometry, priors and status. The original 40 regional trial attempts remain a development record; revising an eruption's source data does not turn those trials into new independent validation. Failed attempts and undercoverage must remain visible.

`python -m pytest -q` and `python scripts/check_repository_navigation.py` check the current code and navigation. `python -m benchmarks.verify_aso_evidence` verifies the published Aso-4 data, matched folds, posterior checksums and retained reference without rewriting its report. [Reproduction commands](../benchmarks/README.md).

Historical test counts and scientific statements describe the runs that produced them. [Older studies in Git history](HISTORY.md) are not the default evidence for current method performance.
