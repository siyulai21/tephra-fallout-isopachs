"""Response-independent local interpolation footprints, with holes preserved.

This is a declared geometric screen, NOT a confidence set or geological boundary.
It must never turn absent sampling into absent fallout. Three fixed policies
expose sensitivity to its non-universal spacing assumptions.
"""
from dataclasses import asdict, dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import Delaunay, cKDTree
from shapely import Polygon, constrained_delaunay_triangles, union_all


@dataclass(frozen=True)
class SupportPolicy:
    name: str = "standard"
    spacing_factor: float = 2.
    deletion_shape_tolerance: float = .25
    grid_shape_tolerance: float = .01

    def __post_init__(self):
        if (not np.isfinite(self.spacing_factor) or self.spacing_factor <= 0
                or not 0 < self.deletion_shape_tolerance < 1
                or not 0 < self.grid_shape_tolerance < 1):
            raise ValueError("Support tolerances must be finite and positive.")


POLICIES = (SupportPolicy("strict", 1.5, .1), SupportPolicy(),
            SupportPolicy("permissive", 3., .5))


class TriangleRegion:
    """Positive quadrature over a polygon union; holes never get filled."""
    def __init__(self, triangles):
        self.triangles = np.asarray(triangles, float).reshape(-1, 3, 2)
        if not np.isfinite(self.triangles).all():
            raise ValueError("Triangle vertices must be finite.")
        ab = self.triangles[:, 1]-self.triangles[:, 0]
        ac = self.triangles[:, 2]-self.triangles[:, 0]
        self.jacobians = abs(ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0])
        self.area = float(self.jacobians.sum()/2)
        self.geometry = union_all([Polygon(t) for t in self.triangles])
        if abs(self.geometry.area-self.area) > 1e-8*max(self.area, 1):
            raise ValueError("Triangle interiors must not overlap.")

    @classmethod
    def from_polygon(cls, geometry):
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Integration polygon must be nonempty and valid.")
        pieces = constrained_delaunay_triangles(geometry)
        region = cls([np.asarray(p.exterior.coords)[:3] for p in pieces.geoms])
        if geometry.symmetric_difference(region.geometry).area > 1e-8*geometry.area:
            raise RuntimeError("Constrained triangulation did not preserve the integration polygon.")
        return region

    def quadrature(self, order):
        if not isinstance(order, int) or order < 2 or self.area <= 0:
            raise ValueError("Need positive area and integer quadrature order >=2.")
        nodes, weights = leggauss(order)
        nodes, weights = (nodes+1)/2, weights/2
        u, v = np.meshgrid(nodes, nodes)
        u, v = u.ravel(), v.ravel()
        w = np.outer(weights, weights).ravel()*(1-u)
        a, b, c = self.triangles.transpose(1, 0, 2)
        points = (a[:, None, :]+u[None, :, None]*(b-a)[:, None, :]
                  +((1-u)*v)[None, :, None]*(c-a)[:, None, :])
        return points.reshape(-1, 2), (self.jacobians[:, None]*w).ravel()


class SupportMesh(TriangleRegion):
    def __init__(self, data, policy=SupportPolicy()):
        # Bounds inform the likelihood, but are not two-sided thickness samples.
        # Duplicate records cannot increase geometric support.
        self.xy = xy = np.unique(data.xy[~data.censored], axis=0)
        if len(xy) < 4 or np.linalg.matrix_rank(xy-xy.mean(axis=0)) < 2:
            raise ValueError("Local support needs four distinct non-collinear exact sites.")
        self.policy = policy
        self.site_scales = cKDTree(xy).query(xy, k=4)[0][:, 3]
        self.triangulation = Delaunay(xy)
        vertices = self.triangulation.simplices
        triangles = xy[vertices]
        sides = np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2)
        ab, ac = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        area2 = abs(ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0])
        # Circumdiameter = a*b*c/(2*area); rejects long, very thin triangles.
        diameter = np.prod(sides, axis=1)/area2
        local = self.site_scales[vertices]
        edge_ratio = sides.max(axis=1)/local.min(axis=1)
        diameter_ratio = diameter/np.median(local, axis=1)
        self.ratios = np.maximum(edge_ratio, diameter_ratio)
        self.accepted = self.ratios <= policy.spacing_factor
        super().__init__(triangles[self.accepted])

    def describe(self):
        return {"policy": asdict(self.policy), "area_km2": self.area,
                "n_exact_distinct_sites": len(self.xy),
                "n_triangles": len(self.accepted), "n_retained_triangles": int(self.accepted.sum()),
                "site_scale_km": self.site_scales.tolist(), "site_xy_km": self.xy.tolist(),
                "triangle_vertices": self.triangulation.simplices.tolist(),
                "triangle_spacing_ratio": self.ratios.tolist(), "retained": self.accepted.tolist(),
                "definition": "Exact-site Delaunay triangles: longest edge / minimum vertex third-neighbor distance and circumdiameter / median vertex third-neighbor distance both <= spacing factor.",
                "interpretation": "Geometry-screened interpolation footprint, not a confidence region, deposit boundary, or guarantee of representativeness. Bounds alone do not establish two-sided local support."}
