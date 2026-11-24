"""Directional continuation with reduced range away from sampled bearings.

The geometry preserves the published likelihood and all measured triangle values. The
range ratio controls a continuation assumption, never the existence or amount of ash.
"""
import numpy as np

from ...core.observations import Observations
from .ensemble.fields import Member
from .geometry import TAU, circular_distance, polar
from .integration import integrate, make_rule

RATIOS = (1., .5, .25, .125)


class BearingGeometry:
    def __init__(self, base, observations, ratio):
        if not 0 < ratio <= 1:
            raise ValueError("Cross-bearing range ratio must be in (0,1]")
        self.base = base
        self.ratio = float(ratio)
        r, a = polar(observations.xy, base.axis)
        self.bearings = np.unique(a[r > 1e-9])
        if not len(self.bearings):
            raise ValueError("At least one quantitative location away from the vent is needed")

    def __getattr__(self, name):
        return getattr(self.base, name)

    def nearest_angle(self, angles):
        angles = np.asarray(angles) % TAU
        ix = np.searchsorted(self.bearings, angles)
        left = self.bearings[(ix - 1) % len(self.bearings)]
        right = self.bearings[ix % len(self.bearings)]
        return np.minimum(circular_distance(angles, left), circular_distance(angles, right))

    def stretch(self, angles):
        delta = np.minimum(self.nearest_angle(angles), np.pi / 2)
        return np.sqrt(1. + (1. / self.ratio**2 - 1.) * np.sin(delta)**2)

    def coords(self, xy):
        radius, angular_weights = self.base.coords(xy)
        if self.ratio == 1.:
            return radius, angular_weights
        _, bearing = polar(xy, self.axis)
        return radius * self.stretch(bearing), angular_weights

    def describe(self):
        return {**self.base.describe(), "cross_bearing_range_ratio": self.ratio,
                "quantitative_bearings_rad": self.bearings.tolist()}

    def quadrature(self, angular, radial):
        rule = make_rule(self.base, angular, radial)
        stretch = self.stretch(rule["angles"])
        g = stretch[rule["gap_ray"]]
        rule["gap_lo"] = g * rule["gap_lo"]
        rule["gap_hi"] = g * rule["gap_hi"]
        rule["gap_aw"] = rule["gap_aw"] / g**2
        rule["local"]["r"] = rule["local"]["r"] * stretch[rule["local"]["ray"]]
        return rule


def with_ratio(member, observations, ratio):
    return Member(BearingGeometry(member.geometry, observations, ratio), member.draws, member.index, member.records)


def complete_masses(draws, geometry):
    """Integrate actual projected area through the antipode, retaining all chain/sector draws."""
    previous = None
    history = []
    actual_area = np.array([geometry.mesh.area, geometry.area[~geometry.inside].sum()])
    for angular, radial in ((1,3),(2,5),(4,8),(8,12),(12,18),(20,24)):
        rule = geometry.quadrature(angular, radial)
        mass = integrate(draws, geometry.base, rule)
        total = mass.sum(axis=(-1,-2))
        if np.any(total <= 0) or not np.isfinite(mass).all() or np.any(mass < 0):
            raise ArithmeticError("Nonfinite or negative directional mass")
        err = float(np.max(abs(rule["area"].sum(1) - actual_area) / np.maximum(actual_area, 1.)))
        row = {"angular_order":angular, "radial_order":radial, "area_relative_error":err}
        history.append(row)
        if previous is not None:
            change = float(np.max(abs(mass - previous).sum(axis=(-1,-2)) / total))
            row["max_partition_L1_change_over_total"] = change
            if change < .005 and err < .001:
                break
        previous = mass
    return mass, {"passed":bool(change < .005 and err < .001), "history":history}


def include_visibility_readings(data, xy, effective_limits_cm):
    """Add conditional archive likelihood terms. Geometry must still use the original quantitative data."""
    xy = np.asarray(xy, dtype=float).reshape(-1,2)
    limits = np.asarray(effective_limits_cm, dtype=float)
    if limits.shape != (len(xy),) or np.any(limits <= 0) or not np.isfinite(limits).all():
        raise ValueError("Supply one positive effective visibility scale per conditional reading")
    if not len(xy):
        return data
    ids = -1 - np.arange(len(xy))
    if np.intersect1d(data.row_ids, ids).size:
        raise ValueError("Archive likelihood IDs overlap quantitative row IDs")
    return Observations(np.vstack((data.xy,xy)), np.r_[data.log_value,np.log(limits)],
                        np.r_[data.censored,np.ones(len(xy),bool)], np.r_[data.log_sd,np.zeros(len(xy))],
                        np.r_[data.row_ids,ids], np.r_[data.lower_censored,np.zeros(len(xy),bool)],
                        np.r_[data.interval_log_upper, np.full(len(xy), np.nan)])
