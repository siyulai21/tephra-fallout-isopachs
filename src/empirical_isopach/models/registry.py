"""The active empirical families. Order is for presentation, never preference.

A fitter accepts (observations, bounds, folds, buffer_km) and returns a fitted
surface plus an auditable selection dictionary. See docs/ADDING_METHOD.md.
"""
from dataclasses import dataclass
from typing import Callable

from .gaussian_process.adapter import fit as fit_gp
from .cubic_spline.adapter import fit as fit_spline, diagnose as diagnose_spline


@dataclass(frozen=True)
class Method:
    label: str
    fit: Callable
    description: str
    diagnose: Callable | None = None


REGISTRY = {
    "stacked": Method("GP ensemble", fit_gp,
        "Predictive stacking of Gaussian-process components; no hand-placed pseudo-data."),
    "spline": Method("Trend-regularized cubic spline", fit_spline,
        "Cubic basis with automatically selected trend and derivative penalties.", diagnose_spline),
}


def active_methods():
    return tuple(REGISTRY)


def method_spec(key):
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown empirical method: {key}. Available: {', '.join(REGISTRY)}") from None


def method_label(key):
    return method_spec(key).label


def method_metadata():
    return {key: {"label": spec.label, "description": spec.description}
            for key, spec in REGISTRY.items()}
