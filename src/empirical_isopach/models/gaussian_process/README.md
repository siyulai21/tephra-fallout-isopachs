# Gaussian-process ensemble implementation

[Families](../README.md) · [Method card](../../../../methods/gaussian-process.md)

- [adapter.py](adapter.py): exposes the ensemble as one active regional method.
- [model.py](model.py): directional trends, Matérn residuals and conditional/censored fitting.
- [ensemble.py](ensemble.py): predictive stacking and mixture behavior.
- [selection.py](selection.py): candidate evaluation, weights and historical selected-GP support.

The original `src.probabilistic_isopach` import paths re-export these implementations. Individual GP components remain necessary to fit the ensemble; the registry does not expose them as separate families.
