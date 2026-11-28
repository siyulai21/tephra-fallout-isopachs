# Wrapped angular covariance implementation

| File | Responsibility |
|---|---|
| [geometry.py](geometry.py) | Spherical transport mean, periodic spectrum, origin and antipode limits |
| [regional.py](regional.py) and [conditioning.py](conditioning.py) | Gaussian point likelihood, conditional bounds, posterior coefficients and preserved-bed predictions |
| [source.py](source.py), [source_prior.py](source_prior.py) and [fitting.py](fitting.py) | Independent source area, conditional peak, near-source observation law and source-prior integration |
| [field.py](field.py) and [packet.py](packet.py) | Arbitrary-location quantiles, spatially coherent coefficient draws and portable posterior |
| [integrate.py](integrate.py) | Whole-Earth pointwise quartiles, joint field volumes and quadrature refinement |
| [validation.py](validation.py) and [report.py](report.py) | Buffered outer folds, preserved-bed forecasts, archive comparisons, shared contour audit and public runner |

[Method and equations](../../../../docs/methods/wrapped-angular-covariance.md) · [Nine-eruption comparison](../../../../comparisons/wrapped-angular-covariance/evidence.md)
