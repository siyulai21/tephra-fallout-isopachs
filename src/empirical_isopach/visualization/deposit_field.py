"""Read principal posterior packets and sample their median thickness over the WGS84 Earth."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.probabilistic_isopach.global_volume import EarthDomain


FAMILIES = {
    "directional": "Directional ensemble",
    "spherical-mixture": "Spherical mixture",
    "nested-isopachs": "Bayesian nested corridors",
}
FIRST_CM = 0.5


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def parent_view(folder, family, data, meta):
    """Return the published median evaluator, support model and source identities."""
    folder = Path(folder)
    if family == "directional":
        from ..models.directional.azimuth.primary import read_field
        from ..models.directional.ensemble.maps import MapView
        paths = ["azimuth/report.json", "azimuth/fields.npz"]
    elif family == "spherical-mixture":
        from ..models.mass_mixture.bearings.field import PROTOCOL
        from ..models.mass_mixture.bearings.maps import BearingView
        paths = ["report.json", "posterior.npz"]
    elif family == "nested-isopachs":
        from ..models.nested.reach import ReachDraws, PROTOCOL
        from ..models.nested.report import MapView
        paths = ["report.json", "full.npz"]
    else:
        raise ValueError(f"Unknown global family {family}")
    report = json.loads((folder/paths[0]).read_text())
    identity = {path:digest(folder/path) for path in paths}
    source = report.get("input", report)
    if source["input_sha256"] != meta["input_sha256"]:
        raise ValueError(f"Accepted data differ from the {family} fit")
    lat = meta["vent_lonlat"][1]
    if family == "directional":
        if report["protocol"] != "uncertain-azimuth-v1":
            raise ValueError("Expected the quantitative uncertain-azimuth field")
        model = read_field(folder/"azimuth", data, report)
        packet = folder/"azimuth/interpolation-mesh.npz"
        if not packet.is_file() and not (folder/"multiscale/report.json").is_file():
            packet=folder/"median-grid.npz"
        if packet.is_file():
            restore_directional_mesh(packet,model,meta["input_sha256"],identity["azimuth/fields.npz"])
        view = MapView(model, lat)
    elif family == "spherical-mixture":
        if report["protocol"] != PROTOCOL or identity[paths[1]] != report["fits"]["baseline"]["posterior_sha256"]:
            raise ValueError("Expected the principal bearing-conditioned posterior")
        with np.load(folder/paths[1]) as packet:
            view = BearingView(dict(packet), data, lat, report["bearing"]["map_draw_check"]["draws_per_chain"])
        model = None
    else:
        if report["protocol"] != PROTOCOL or identity[paths[1]] != report["full_packet_sha256"]:
            raise ValueError("Expected the principal nested corridor posterior")
        model = ReachDraws.load(folder/paths[1])
        view = MapView(model.reduced(report["quadrature"][-1]["draws"]))
    return view, model, identity


def review_path(folder, family):
    smooth = Path(folder)/"multiscale/report.json"
    if family == "directional" and smooth.is_file():
        return smooth
    footprints = Path(folder)/"footprints/report.json"
    if family == "nested-isopachs" and footprints.is_file():
        return footprints
    return Path(folder)/"extent/report.json"


def display_name(family, sources):
    if "multiscale/report.json" in sources:
        return "Multiscale directional ensemble"
    return "Bayesian nested footprints" if "footprints/report.json" in sources else FAMILIES[family]


def open_field(folder, family, data, meta, *, review=None):
    """Prefer the current complete-deposit model when a reviewed continuation exists."""
    path = Path(review) if review is not None else review_path(folder,family)
    if family == "directional" and path.parent.name == "multiscale" and path.is_file():
        from ..models.directional.multiscale.geometry import SCHEMA, LEGACY_SCHEMA
        from ..models.directional.multiscale.posterior import DirectionalField
        report=json.loads(path.read_text())
        packet=path.parent/"posterior.npz"
        if (report["protocol"] not in (SCHEMA, LEGACY_SCHEMA) or report["input_sha256"]!=meta["input_sha256"]
                or digest(packet)!=report["artifacts_sha256"]["posterior.npz"]):
            raise ValueError("Directional data or posterior identity differs")
        model=DirectionalField.load(packet)
        return model,model,{"multiscale/report.json":digest(path),"multiscale/posterior.npz":digest(packet)}
    if family == "nested-isopachs" and path.parent.name == "footprints" and path.is_file():
        from ..models.nested.footprints.posterior import FootprintField, PROTOCOL as FOOTPRINTS, LEGACY_PROTOCOL, GROWTH_PROTOCOL
        from ..models.nested.footprints.continuous import FlowField, PROTOCOL as CONTINUOUS
        from ..models.nested.report import MapView
        report = json.loads(path.read_text())
        packet = path.parent/"full-selected.npz"
        key = "footprints/full-selected.npz"
        if (report["protocol"] not in (FOOTPRINTS, LEGACY_PROTOCOL, GROWTH_PROTOCOL, CONTINUOUS) or report["input_sha256"] != meta["input_sha256"] or
                digest(packet) != report["posterior_sha256"][packet.name]):
            raise ValueError("Nested footprints belong to different data or posterior samples")
        model = (FlowField if report["protocol"]==CONTINUOUS else FootprintField).load(packet)
        return MapView(model.reduced(report["map_draws"])), model, {"footprints/report.json":digest(path), key:digest(packet)}
    view, model, sources = parent_view(folder, family, data, meta)
    if not path.is_file():
        return view, model, sources
    from ..models.extent_review import ExtentFields, ExtentView, PROTOCOL
    report = json.loads(path.read_text())
    if (report["protocol"]!=PROTOCOL or report["family"]!=family or report["input_sha256"]!=meta["input_sha256"]
            or report["parent_sha256"]!=sources):
        raise ValueError("Conditional continuation belongs to different input or posterior")
    cal = report["calibration"]
    if not report["exact_parent_identity"]:
        parent = view.field if family=="spherical-mixture" else view.model
        fields = ExtentFields(family,parent,data,meta["vent_lonlat"][1],cal["states"])
        view = ExtentView(fields,view.domain,cal["weights"])
    return view,model,{**sources,"extent/report.json":digest(path)}


@dataclass
class PolarField:
    """Periodic field on equal-area chart radii and uniformly spaced bearings clockwise from north."""
    radii: np.ndarray
    log10_cm: np.ndarray
    axis: float

    @property
    def angles(self):
        return np.arange(self.log10_cm.shape[1]) * (2*np.pi/self.log10_cm.shape[1])

    def outer_crossings(self, cm):
        """Outermost exceedance on each bearing, retaining detections beyond gaps."""
        t = np.log10(cm)
        high = self.log10_cm >= t
        n = len(self.radii)
        last = n - 1 - high[::-1].argmax(axis=0)
        j = np.arange(len(last))
        inside = high.any(axis=0)
        result = np.full(len(j), np.nan)
        rim = inside & (last == n-1)
        result[rim] = self.radii[-1]
        k = j[inside & ~rim]
        i = last[k]
        frac = (t - self.log10_cm[i,k]) / (self.log10_cm[i+1,k] - self.log10_cm[i,k])
        result[k] = self.radii[i] + frac * (self.radii[i+1] - self.radii[i])
        return result

    def area(self, cm, *, stride=1, phase=0):
        """Sum every radial interval above threshold, including detached rings. Chart area is physical km²."""
        indices = np.unique(np.r_[np.arange(0, len(self.radii), stride), len(self.radii)-1])
        r = self.radii[indices, None]
        z = self.log10_cm[indices, phase::stride]
        a, b = z[:-1], z[1:]
        t = np.log10(cm)
        above_a, above_b = a >= t, b >= t
        split = above_a != above_b
        frac = np.divide(t-a, b-a, out=np.zeros_like(a), where=split)
        cut = r[:-1] + frac * np.diff(r, axis=0)
        annulus = np.where(above_a & above_b, np.diff(r**2, axis=0), 0.)
        annulus += np.where(split, np.where(above_a, cut**2-r[:-1]**2, r[1:]**2-cut**2), 0.)
        return float(np.pi * annulus.sum() / z.shape[1])

    def at_xy(self, x, y):
        """Evaluate the periodic log-thickness interpolant at paired chart coordinates."""
        xx, yy = np.asarray(x)/self.axis, np.asarray(y)*self.axis
        ang = np.append(self.angles, 2*np.pi)
        z = np.column_stack((self.log10_cm, self.log10_cm[:, 0]))
        interpolator = RegularGridInterpolator((self.radii, ang), z, bounds_error=False, fill_value=np.nan)
        return interpolator((np.hypot(xx, yy), np.arctan2(xx, yy) % (2*np.pi)))


    def on_grid(self, x, y):
        return self.at_xy(*np.meshgrid(x, y))


def write_map_grid(path,field,model,family,meta,sources):
    """Preserve the actual chart and the directional mesh used to interpolate posterior nodes."""
    arrays = dict(radii_km=field.radii, log10_cm=field.log10_cm, axis=field.axis,
                  input_sha256=meta["input_sha256"], sources_json=json.dumps(sources,sort_keys=True))
    if family=="directional" and "multiscale/report.json" not in sources:
        g = model.members[0].geometry
        arrays.update(triangles_km=g.triangles, local_vertex_ids=g.node_ids, vertices_km=g.vertices,
                      inside_observation_mesh=g.inside)
    np.savez_compressed(path,**arrays)


def restore_directional_mesh(path,model,input_hash,posterior_hash):
    """Reuse the published triangle connectivity across GEOS and coordinate-library builds."""
    import shapely
    g = model.members[0].geometry
    with np.load(path,allow_pickle=False) as packet:
        if (str(packet["input_sha256"])!=input_hash or
                json.loads(str(packet["sources_json"]))["azimuth/fields.npz"]!=posterior_hash):
            raise ValueError("The saved interpolation mesh belongs to a different directional fit")
        vertices = packet["vertices_km"]
        np.testing.assert_allclose(vertices,g.vertices,rtol=0,atol=1e-7)
        triangles = packet["triangles_km"]
        ids = packet["local_vertex_ids"]
        if ids.shape!=triangles.shape[:2] or np.any((ids < -1)|(ids>=len(vertices))):
            raise ValueError("Invalid vertex index in the directional interpolation mesh")
        np.testing.assert_allclose(triangles[ids>=0],vertices[ids[ids>=0]],rtol=0,atol=1e-9)
        g.inside = packet["inside_observation_mesh"]
    g.triangles,g.node_ids = triangles,ids
    g.polygons = shapely.polygons(triangles)
    g.tree = shapely.STRtree(g.polygons)
    g.area = shapely.area(g.polygons)
    g.inverse = np.linalg.inv(np.swapaxes(triangles[:,1:]-triangles[:,:1],1,2))


def polar_nodes(data, latitude, n_radii, n_angles):
    """Log-spaced radial profiles cover the centre through the antipode and resolve the near vent."""
    domain = EarthDomain.wgs84(latitude)
    positive = domain.rho(data.xy)
    off_vent = positive[positive>0]
    first = np.clip(off_vent.min()/16,.01,1.) if len(off_vent) else .01
    radii = np.r_[0., np.geomspace(first, domain.rho_max, n_radii-1)]
    # The antipode is a coordinate singularity. Sample its interior limit and assign it to the rim.
    rr = radii[1:].copy()
    rr[-1] *= 1.-1e-8
    theta = np.arange(n_angles) * (2*np.pi/n_angles)
    xy = np.stack((np.outer(rr, np.sin(theta))*domain.axis_scale,
                   np.outer(rr, np.cos(theta))/domain.axis_scale), axis=-1)
    return radii, xy.reshape(-1,2), domain.axis_scale


def sample_field(view, data, latitude, n_radii=193, n_angles=240):
    radii, points, axis = polar_nodes(data, latitude, n_radii, n_angles)
    z = np.r_[np.repeat(view.central(np.zeros((1,2))), n_angles), view.central(points)]
    if not np.isfinite(z).all():
        raise ValueError("The physical Earth contains nonfinite median thicknesses")
    return PolarField(radii, z.reshape(n_radii,n_angles), axis)


def doubling_levels(max_cm):
    if not np.isfinite(max_cm) or max_cm <= 0:
        raise ValueError("Thickness must be positive and finite")
    return FIRST_CM * 2.**np.arange(max(0, np.floor(np.log2(max_cm/FIRST_CM)+1e-12))+1)
