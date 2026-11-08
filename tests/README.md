# Verification tests

[Project home](../README.md) · [Validation guide](../docs/VALIDATION.md) · [Contributing](../CONTRIBUTING.md)

Run `python -m pytest -q` after installing `.[test]`. These tests verify numerical and workflow behavior; they do not establish the true unobserved volume of a real eruption.

| Area | Tests |
|---|---|
| Active regional models and probability | `test_probabilistic.py`, `test_ensemble.py`, `test_regional_spline.py` |
| Global posterior, normalization and geometry | `test_mass_mixture.py`, `test_source_fans.py`, `test_vent_envelope.py` |
| Local footprints and complete-contour diagnostics | `test_support.py` |
| Source audits and preserved inputs | `test_aso_evidence.py`, `test_minoan.py`, `test_ytt.py` |
| Reporting and future-family integration | `test_reporting.py`, `test_method_registry.py` |
| Historical numerical and import compatibility | `test_global_volume.py`, `test_spline_numerics.py`, `test_spline_prior.py` |

The registry test exercises a test-only third field through common fitting, folds, integration and reports; it does not register a third regional scientific model. The small [YTT fixture](fixtures/README.md) preserves an existing regression reference without retaining its full unused report collection.

`python scripts/check_repository_navigation.py` checks repository-local Markdown links and images. `python -m benchmarks.verify_aso_evidence` verifies source/artifact/posterior checksums and matched geographic folds. GitHub Actions also installs the wheel and checks CLI routes, accepted case data and every catalogued audit outside the checkout.

Known-truth stress tests and publication findings live in the [evidence catalogue](../comparisons/README.md). Historical counts describe their original executions; use the current test run for the current code.
