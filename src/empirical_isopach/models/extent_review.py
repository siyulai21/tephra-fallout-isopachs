"""Coherent alternatives for the three existing global deposit families."""
import numpy as np

from .continuation_geometry import SampleLayout
from .directional.azimuth.resolution import resolve
from .directional.ensemble.fields import inverse_cdf
from .mass_mixture.bearings.field import POWERS, multiplier
from .mass_mixture.bearings.flank import FlankKernels
from .nested.spreading import SampledRidge, field_difference

STATES = ("parent", "broad", "resolved")
FAMILY_NAMES = ("directional", "spherical-mixture", "nested-isopachs")
PROTOCOL = "finite-spread-and-angular-resolution-v1"


class ExtentFields:
    """Each row of each variant is the same complete posterior realization at every position."""
    def __init__(self, family, parent, data, latitude, states=STATES):
        self.family = family
        self.parent = parent
        self.states = tuple(states)
        if not set(self.states) <= set(STATES):
            raise ValueError("Unknown continuation")
        self.layout = SampleLayout(data, latitude)
        if family == "spherical-mixture":
            self.kernels = FlankKernels(parent.draws, data, latitude)
            n = len(parent.phi)*len(POWERS)
            self.weights = np.full(n, 1./n)
        elif family == "nested-isopachs":
            n = len(parent.law)
            self.ridge = SampledRidge(parent, self.layout)
            self.weights = np.full(n, 1./n)
        elif family == "directional":
            self.models = {s:(parent if s=="parent" else resolve(parent, self.layout, 11.25 if s=="broad" else 22.5))
                           for s in self.states}
            self.weights = parent.weights
        else:
            raise ValueError(f"Unknown global family {family}")

    def variants(self, xy):
        p = self.parent
        if self.family == "spherical-mixture":
            base = self.kernels.variants(xy, self.layout, self.states)
            h, theta = p.support.evaluate(xy)
            shifts = [np.log(multiplier(h, theta, p.phi[:, None], k)) for k in POWERS]
            return [np.concatenate([b+shift for shift in shifts]) for b in base]
        if self.family == "nested-isopachs":
            from .nested.geometry import positions
            radius,bearing = positions(xy,p.latitude)
            base = p.parent.field(xy)
            original = base+p.corridors.penalties(radius,bearing)[p.law]
            delta = field_difference(p, self.layout, xy)[p.law]
            excess = np.maximum(base-self.ridge.flank(radius),0.)
            outer = self.layout.lengths.max()
            leverage = 1./(1.+(np.maximum(radius-outer,0.)/outer)**2)
            delta *= leverage
            free = self.layout.freedom(xy)*leverage*(p.law[:,None]!=0)
            outputs = []
            for state in self.states:
                if state=="parent":
                    outputs.append(original)
                    continue
                b = np.log(4. if state=="broad" else 2.)
                loss = excess-b*np.tanh(excess/b)
                outputs.append(original+(.5 if state=="broad" else 1.)*delta-free*loss)
            return outputs
        return [m.distribution(xy).values for m in self.models.values()]


class ExtentView:
    def __init__(self, fields, domain, mixture):
        self.fields, self.domain = fields, domain
        mix = np.asarray(mixture, dtype=float)
        if len(mix)!=len(fields.states) or not np.isclose(mix.sum(),1) or np.min(mix)<0:
            raise ValueError("Invalid continuation weights")
        self.weights = np.concatenate([w*fields.weights for w in mix])
        self.equal = np.allclose(self.weights, self.weights[0], rtol=1e-12, atol=0.)

    def central(self, xy):
        xy = np.asarray(xy)
        z = np.full(len(xy), np.nan)
        valid = np.flatnonzero(self.domain.rho(xy)<self.domain.rho_max)
        for i in range(0, len(valid), 192):
            idx = valid[i:i+192]
            a = np.concatenate(self.fields.variants(xy[idx]))
            median = np.median(a, axis=0) if self.equal else inverse_cdf(a.T, self.weights)[0]
            z[idx] = median/np.log(10.)
        return z
