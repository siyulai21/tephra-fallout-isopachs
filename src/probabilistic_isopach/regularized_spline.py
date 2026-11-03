"""Compatibility imports for the experimental trend-regularized cubic spline.

Implementation: src.research.trend_regularized_spline.
The Gaussian-process report workflow does not call this module.
"""
from src.research.trend_regularized_spline import (
    RegularizedSpline, design_matrix, select_spline_prior,
)

__all__ = ["RegularizedSpline", "design_matrix", "select_spline_prior"]
