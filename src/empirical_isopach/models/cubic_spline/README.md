# Automatic cubic-spline implementation

[Families](../README.md) · [Method card](../../../../methods/cubic-spline.md)

- [model.py](model.py): cubic tensor basis, normalized penalties, mixed likelihood and conditional coefficient uncertainty.
- [selection.py](selection.py): geographic tuning and bounded search expansion.
- [adapter.py](adapter.py): shared workflow entry, penalty-quadrature check and spline-specific reliability notes.

This is the active trend-regularized spline. It does not fit a GP residual. The old `src.empirical_isopach.spline` and `selection` modules re-export it; the original historical spline is a separate compatibility package.
