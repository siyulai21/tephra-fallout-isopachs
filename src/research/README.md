# Historical spline research

[Source map](../README.md) · [Method comparison](../../docs/METHODS.md) · [Mathematical description](../../docs/methods/cubic-splines.md)

**For current fitting, use the [automatic spline and regional comparison](../empirical_isopach/README.md).**

`trend_regularized_spline.py` preserves the **first trend-regularized spline experiment**. It retains the original basis and tension/roughness penalties, then adds an area-averaged penalty toward a thinning trend estimated from measured data.

Numerical evaluation points approximate one normalized penalty rather than supplying additional independent measurements. Held-out data select its strength, including zero. Existing tension/roughness defaults remain fixed during that selection.

**Research status:** point estimates only; no current eruption report uses this method. It rejects censored observations and predictions outside its fitted region. The trend comes from the same observations, so this is data-adaptive regularization rather than independent prior evidence.

```python
from src.research.trend_regularized_spline import RegularizedSpline, select_spline_prior
```

The previous import from `src.probabilistic_isopach.regularized_spline` remains compatible. The [initial benchmark](../../archive/validation/initial-benchmark.md) records what was evaluated, including boundary optima and the absence of independent nested evaluation for the spline-prior selection.
