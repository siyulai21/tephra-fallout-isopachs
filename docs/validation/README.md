# Validation and research benchmarks

[Aso-4 source and observation-process tests](../../comparisons/aso-4-audit/README.md) include matched geometry comparisons, exact deletion refits, explicit marine-setting stress tests and retained source conflicts.

[Documentation](../README.md) · [Method comparison](../METHODS.md)

Read each result with its execution date, input selection and model scope. Historical test counts and regional volumes are records of those runs, not statements about the latest release.

| Evidence | Scope | Where to read it |
|---|---|---|
| Current matched regional comparison | Six eruptions, exhaustive spline tuning, common nested holdouts and known-truth trials | [Peer-method evidence](../../comparisons/regional/README.md) |
| Earlier complete-Earth integration and six-eruption comparison | GP selected/stacked fits, numerical checks, geographic deletions and synthetic trials | [Historical analysis](../../archive/reports/global-fallout/README.md) and [verification](../../archive/reports/global-fallout/validation.json) |
| Current eruption predictions and assumptions | Paired GP-ensemble / spline calculations with input/code hashes | [Eruption catalogue](../../eruptions/README.md) |
| Initial 1 November 2026 research benchmark | Four eruptions; GP, fixed-default original spline and separate spline-prior selection diagnostics | [Report](../../archive/validation/initial-benchmark.md), [numerical record](../../archive/validation/initial-benchmark.json) |
| GP ensemble development benchmark | Four eruptions and 30 synthetic trials; selected versus stacked GP | [Method and findings](../methods/gp-ensemble.md), [artifacts](../../archive/validation/ensemble/README.md) |
| Numerical/statistical tests | Checks for the installed code version | [Test guide](../../tests/README.md) |

The original-spline baseline was not independently tuned against every GP configuration. Spline-prior selection scores were not independent outer validation. These development experiments are not proof of universal method superiority or independent truth for real eruption volumes.

Numerical files are preserved unchanged. Their source paths and hashes describe the original run. Historical artifacts remain unchanged; new matched comparisons have their own source hashes and explicit regional scope.
