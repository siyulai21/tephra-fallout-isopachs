"""Training-only geometry for a continuous whole-Earth thickness field."""
from dataclasses import asdict, dataclass, replace

import numpy as np
import shapely
from scipy.spatial import cKDTree

from ...support import SupportMesh, SupportPolicy
from ..mass_mixture.geometry import RADIUS
from src.probabilistic_isopach.global_volume import EarthDomain


@dataclass(frozen=True)
class Policy:
    name: str = "reference"
    fixed_tail_shape: float | None = None
    half_width_deg: float = 45.
    spacing_factor: float = 2.
    collar_spacing: float = 1.
    local_sd_ratio: float = 1.
    background_scale_km: float = 35.
    hidden_fan: bool = False


BASE = Policy()
POLICIES = (BASE, replace(BASE, name="heavy_tail", fixed_tail_shape=.5),
            replace(BASE, name="fixed_fourth_power", fixed_tail_shape=.25),
            replace(BASE, name="exponential", fixed_tail_shape=0.),
            replace(BASE, name="light_tail", fixed_tail_shape=.125),
            replace(BASE, name="narrow_fans", half_width_deg=22.5),
            replace(BASE, name="broad_fans", half_width_deg=90.),
            replace(BASE, name="possible_missing_fan", hidden_fan=True),
            replace(BASE, name="strict_footprint", spacing_factor=1.5),
            replace(BASE, name="permissive_footprint", spacing_factor=3.),
            replace(BASE, name="broader_background", background_scale_km=105.),
            replace(BASE, name="longer_transition", collar_spacing=2.),
            replace(BASE, name="stronger_local_departures", local_sd_ratio=2.))
COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
TAU = 2*np.pi


def polar(xy, axis):
    """Source-centred authalic chart, compass bearing clockwise from north."""
    p = np.asarray(xy) * [1/axis, axis]
    return np.linalg.norm(p, axis=-1), np.arctan2(p[..., 0], p[..., 1]) % TAU


def circular_distance(first, second):
    return np.abs((first-second+np.pi) % TAU-np.pi)


def angular_weights(bearing, centers, width):
    distance = circular_distance(np.asarray(bearing)[..., None], centers)
    taper = .5*(1+np.cos(np.pi*np.minimum(distance/width, 1.)))
    return taper / np.maximum(1., taper.sum(axis=-1, keepdims=True))


def fan_design(vertices, exact, axis, width, hidden):
    radii, angles = polar(exact, axis)
    supported = {tuple(point) for point in vertices}
    widths = np.array([width if tuple(x) in supported else width/2 for x in exact])
    selected = []
    for i in np.argsort(-radii, kind="stable"):
        if radii[i] <= 0:
            continue
        # A distal outcrop licenses only a narrow directional term when it
        # lacks a two-dimensional interpolation neighbourhood.
        separation = circular_distance(angles[i], angles[selected])
        if not selected or np.all(separation >= np.minimum(widths[i], widths[selected])/2):
            selected.append(i)
    centers = angles[selected]
    reaches = radii[selected]
    half_widths = widths[selected]
    far_angles = angles[radii >= np.median(radii)]
    ordered = np.unique(far_angles)
    gaps = np.diff(np.r_[ordered, ordered[0]+TAU])
    k = int(np.argmax(gaps))
    gap = {"start_deg": float(np.rad2deg(ordered[k])), "width_deg": float(np.rad2deg(gaps[k])),
           "midpoint_deg": float(np.rad2deg((ordered[k]+gaps[k]/2) % TAU))}
    if hidden:
        centers = np.r_[centers, np.deg2rad(gap["midpoint_deg"])]
        reaches = np.r_[reaches, np.quantile(radii, .75)]
        half_widths = np.r_[half_widths, width]
    return centers, reaches, half_widths, gap


class Geometry:
    def __init__(self, data, latitude, policy=BASE):
        self.policy = policy
        self.axis = EarthDomain.wgs84(latitude).axis_scale
        self.mesh = mesh = SupportMesh(data, SupportPolicy(spacing_factor=policy.spacing_factor))
        self.vertices = np.unique(mesh.triangles.reshape(-1, 2), axis=0)
        self.centers, self.reaches, self.width, self.gap = fan_design(
            self.vertices, mesh.xy, self.axis, np.deg2rad(policy.half_width_deg), policy.hidden_fan)
        # Bevelled buffer lies within one transition width of the footprint.
        nearest = cKDTree(mesh.xy).query(mesh.xy, k=2)[0][:, 1]
        earth_margin = max(0., (2*RADIUS-polar(mesh.xy, self.axis)[0].max()) /
                           max(self.axis, 1/self.axis))
        self.collar_km = min(np.median(nearest)*policy.collar_spacing, .49*earth_margin)
        expanded = mesh.geometry.buffer(self.collar_km, join_style="bevel")
        collar = expanded.difference(mesh.geometry)
        extra = shapely.constrained_delaunay_triangles(collar)
        triangles = list(mesh.triangles)
        triangles.extend(np.asarray(t.exterior.coords)[:3] for t in extra.geoms if t.area > 0.)
        self.triangles = np.asarray(triangles).reshape(-1, 3, 2)
        self.inside = np.arange(len(triangles)) < len(mesh.triangles)
        self.polygons = shapely.polygons(self.triangles)
        self.tree = shapely.STRtree(self.polygons)
        self.area = shapely.area(self.polygons)
        if abs(self.area.sum()-expanded.area) > 1e-8*max(1., expanded.area):
            raise ValueError("Transition triangulation lost area")
        if len(triangles):
            # Map retained boundary vertices to actual nodes. New outer nodes
            # have zero local weight. Both sides use identical edge values.
            tree = cKDTree(self.vertices)
            dist, ids = tree.query(self.triangles.reshape(-1, 2))
            self.node_ids = np.where(dist <= 1e-9, ids, -1).reshape(-1, 3)
            self.inverse = np.linalg.inv(np.swapaxes(self.triangles[:, 1:]-self.triangles[:, :1], 1, 2))
        else:
            self.node_ids = np.empty((0, 3), int)
            self.inverse = np.empty((0, 2, 2))

    def locate(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        indices = np.full(len(xy), -1, int)
        weights = np.zeros((len(xy), 3))
        ids = np.full((len(xy), 3), -1, int)
        if len(self.triangles):
            queries, triangles = self.tree.query(shapely.points(xy), predicate="intersects")
            indices[queries] = triangles
            hit = indices >= 0
            at = indices[hit]
            uv = np.einsum("nij,nj->ni", self.inverse[at], xy[hit]-self.triangles[at, 0])
            bary = np.clip(np.column_stack((1-uv.sum(1), uv)), 0, 1)
            bary /= bary.sum(axis=1, keepdims=True)
            ids[hit] = self.node_ids[at]
            weights[hit] = np.where(ids[hit] >= 0, bary, 0.)
        return indices, ids, weights

    def coords(self, xy):
        radius, angle = polar(xy, self.axis)
        return radius, angular_weights(angle, self.centers, self.width)

    def describe(self):
        return {"policy": asdict(self.policy), "authalic_axis_scale": self.axis,
                "support": self.mesh.describe(), "transition_width_km": self.collar_km,
                "transition_area_km2": float(self.area[~self.inside].sum()),
                "local_node_xy_km": self.vertices.tolist(),
                "fan_bearings_deg": np.rad2deg(self.centers).tolist(), "reach_km": self.reaches.tolist(),
                "fan_half_widths_deg": np.rad2deg(self.width).tolist(), "distal_gap": self.gap}

