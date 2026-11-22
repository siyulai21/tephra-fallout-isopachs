"""Circular smoothing of the directional range basis outside resolved observations."""
from copy import copy

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..geometry import angular_weights, polar, TAU
from ..ensemble.fields import Member


class ResolvedGeometry:
    def __init__(self, geometry, layout, sigma_deg):
        self.original = geometry
        self.layout = layout
        bearings = np.arange(720)*TAU/720
        raw = angular_weights(bearings, geometry.centers, geometry.width)
        self.angular = gaussian_filter1d(raw, sigma_deg/0.5, axis=0, mode="wrap")
        self.nodes = len(bearings)

    def __getattr__(self, key):
        return getattr(self.original, key)

    def coords(self, xy):
        radius, theta = polar(xy, self.axis)
        original = angular_weights(theta, self.centers, self.width)
        position = theta*self.nodes/TAU
        a = position.astype(int) % self.nodes
        t = (position-np.floor(position))[:, None]
        smooth = (1.-t)*self.angular[a] + t*self.angular[(a+1) % self.nodes]
        distance = self.layout.sites.query(np.asarray(xy)*self.layout.scale)[0]
        radius_scale = self.layout.spacing + radius*np.sin(np.pi/8)
        outer = self.layout.lengths.max()
        radial_leverage = 1./(1.+(np.maximum(radius-outer, 0.)/outer)**2)
        freedom = (self.layout.freedom(xy)*radial_leverage*(-np.expm1(-(distance/radius_scale)**2)))[:, None]
        return radius, original + freedom*(smooth-original)


def resolve(model, layout, sigma_deg=22.5):
    """Use one coherent azimuth and the same posterior draw throughout each deposit."""
    result = copy(model)
    result.members = []
    for member in model.members:
        g = ResolvedGeometry(member.geometry, layout, sigma_deg)
        result.members.append(Member(g, member.draws, member.index, member.records))
    return result
