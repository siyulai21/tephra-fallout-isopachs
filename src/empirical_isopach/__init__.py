"""Peer empirical surface methods with matched regional assessment."""

from .spline import SplineCandidate, TrendRegularizedSpline
from .selection import select_spline

__all__ = ["SplineCandidate", "TrendRegularizedSpline", "select_spline"]
