"""Sampling geometry shared by conditional extensions of existing deposit models."""
import numpy as np
from scipy.spatial import cKDTree
import shapely

from ..support import SupportMesh
from src.probabilistic_isopach.global_volume import EarthDomain


class SampleLayout:
    """Finite positive transects and protected field measurements in the equal-area chart."""
    def __init__(self, data, latitude):
        axis = EarthDomain.wgs84(latitude).axis_scale
        self.scale = np.array([1 / axis, axis])
        xy = data.xy * self.scale
        positive = np.unique(xy[~data.censored | data.lower_censored], axis=0)
        lengths = np.linalg.norm(positive, axis=1)
        self.lengths = lengths[lengths > 0]
        self.directions = positive[lengths > 0] / self.lengths[:, None]
        if not len(self.lengths):
            raise ValueError("A positive site away from the source is required")
        self.proximal = float(np.median(np.sort(self.lengths)[:3]))
        # Traceless second moment of the source-to-station vectors. Reversing a wind axis
        # preserves it, and rotations of the map cannot change its magnitude.
        bearing_vector = self.directions[:,1]+1j*self.directions[:,0]
        self.axial_concentration = float(abs(np.average(bearing_vector**2, weights=self.lengths**2)))
        exact = np.unique(xy[~data.censored], axis=0)
        self.spacing = float(np.median(cKDTree(exact).query(exact, k=2)[0][:, 1]))
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("At least two distinct measured sites are required")
        self.sites = cKDTree(np.unique(xy, axis=0))
        self.footprint = SupportMesh(data).geometry
        self.bounds = None if self.footprint.is_empty else np.array(self.footprint.bounds)
        self.bounds_pad = self.spacing / min(self.scale)
        # Equal-area coordinates and mesh coordinates differ by a constant axis ratio.
        self.mesh_distance_scale = min(self.scale)

    def freedom(self, xy):
        """Smooth release from every training site and every retained triangle, including holes."""
        xy = np.asarray(xy)
        distance = self.sites.query(xy * self.scale)[0]
        if self.bounds is not None:
            inside = np.flatnonzero(((xy >= self.bounds[:2] - self.bounds_pad)
                                     & (xy <= self.bounds[2:] + self.bounds_pad)).all(axis=1))
            if len(inside):
                border = shapely.distance(shapely.points(xy[inside]), self.footprint)
                distance[inside] = np.minimum(distance[inside], border * self.mesh_distance_scale)
        t = np.clip(distance / self.spacing, 0., 1.)
        return t*t*(3.-2.*t)

    def axial_leverage(self, xy):
        radius = np.linalg.norm(np.asarray(xy)*self.scale, axis=1)
        outer = self.lengths.max()
        return self.axial_concentration/(1.+(np.maximum(radius-outer,0.)/outer)**2)

    def costs(self, xy, *, at_survey_edge=False):
        """Finite-track transverse and endpoint distances in units of dispersive reach.

        Variance grows linearly in distance beyond the proximal survey resolution.
        An isolated distal find can establish a long axis without specifying a wide fan.
        """
        arr = np.asarray(xy) * self.scale
        if at_survey_edge:
            radius = np.linalg.norm(arr, axis=1)
            arr = arr*np.minimum(1., self.lengths.max()/np.maximum(radius, 1e-30))[:, None]
        along = arr @ self.directions.T
        cross = arr[:, :1]*self.directions[:, 1] - arr[:, 1:]*self.directions[:, 0]
        projected = np.clip(along, 0., self.lengths)
        s = self.proximal
        across = (cross*cross + np.minimum(along, 0.)**2) / (s*s + 2.*s*projected)
        beyond = np.maximum(along-self.lengths, 0.)**2 / (s*s + 2.*s*self.lengths)
        return across, beyond


def half_thickness(cost):
    """Log multiplier with half thickness at one scaled displacement."""
    return -2.*np.log1p((np.sqrt(2.)-1.)*np.maximum(cost, 0.))
