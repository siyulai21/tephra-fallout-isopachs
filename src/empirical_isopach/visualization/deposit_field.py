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


def open_field(folder, family, data, meta):
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

    def on_grid(self, x, y):
        xx, yy = np.meshgrid(x/self.axis, y*self.axis)
        ang = np.append(self.angles, 2*np.pi)
        z = np.column_stack((self.log10_cm, self.log10_cm[:, 0]))
        interpolator = RegularGridInterpolator((self.radii, ang), z, bounds_error=False, fill_value=np.nan)
        return interpolator((np.hypot(xx, yy), np.arctan2(xx, yy) % (2*np.pi)))


def sample_field(view, data, latitude, n_radii=193, n_angles=240):
    """Log-spaced radial profiles cover the centre through the antipode and resolve the near vent."""
    domain = EarthDomain.wgs84(latitude)
    positive = domain.rho(data.xy)
    first = np.clip(positive[positive > 0].min()/16, .01, 1.)
    radii = np.r_[0., np.geomspace(first, domain.rho_max, n_radii-1)]
    # The antipode is a coordinate singularity. Sample its interior limit and assign it to the rim.
    rr = radii[1:].copy()
    rr[-1] *= 1.-1e-8
    theta = np.arange(n_angles) * (2*np.pi/n_angles)
    xy = np.stack((np.outer(rr, np.sin(theta))*domain.axis_scale,
                   np.outer(rr, np.cos(theta))/domain.axis_scale), axis=-1)
    points = xy.reshape(-1,2)
    z = np.r_[np.repeat(view.central(np.zeros((1,2))), n_angles), view.central(points)]
    if not np.isfinite(z).all():
        raise ValueError("The physical Earth contains nonfinite median thicknesses")
    return PolarField(radii, z.reshape(n_radii,n_angles), domain.axis_scale)


def doubling_levels(max_cm):
    if not np.isfinite(max_cm) or max_cm <= 0:
        raise ValueError("Thickness must be positive and finite")
    return FIRST_CM * 2.**np.arange(max(0, np.floor(np.log2(max_cm/FIRST_CM)+1e-12))+1)
