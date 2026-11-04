"""Expose one GP ensemble to the shared regional workflow."""
from .selection import select_model


def fit(data, bounds, folds, buffer_km):
    return select_model(data, n_folds=folds, buffer_km=buffer_km, method="stacked")
