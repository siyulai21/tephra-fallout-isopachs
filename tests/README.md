# Verification tests

[Project home](../README.md) · [Contributing](../CONTRIBUTING.md) · [Validation evidence](../docs/validation/README.md)

Run `python -m pytest -q` from the repository root after installing `.[test]`.

| Tests | Scope |
|---|---|
| `test_probabilistic.py` | GP conditioning, censoring, spatial selection and invariance |
| `test_ensemble.py` | Stacking and mixture distribution semantics |
| `test_global_volume.py` | Global geometry, integration, tail diagnostics and domain independence |
| `test_minoan.py`, `test_ytt.py` | Audit reconstruction, input semantics and published-regression checks |
| `test_reporting.py` | Contour geometry, exports and report behavior |
| `test_spline_numerics.py` | Original cubic-spline basis, system and numerical solvers |
| `test_spline_prior.py` | Research spline penalty, normalization and compatibility imports |

Passing tests does not validate a real global eruption volume. Predictive failures and extrapolation assumptions remain in the reports. Test counts in historical validation files refer to those executions.

`python scripts/check_repository_navigation.py` checks local Markdown links and images. GitHub Actions also installs the package outside the checkout to catch missing bundled inputs or import problems.
