"""Finite cross-track continuation of the displaced spherical kernels."""
import numpy as np

from scipy.special import logsumexp
from scipy.spatial import cKDTree

from ...continuation_geometry import half_thickness
from ..geometry import centers, to_sphere, log_field, log_kernel


class FlankKernels:
    """Limit the inferred rise above a sampled flank, component by component."""
    def __init__(self, draws, data, latitude):
        self.draws = draws
        self.latitude = latitude
        self.displaced = "angular" not in draws and "core_log_scale" not in draws
        if self.displaced:
            self.centers = centers(draws["log_scale"], draws["offset"])
            sites = np.vstack(([0., 0.], data.xy[~data.censored | data.lower_censored]))
            unit = to_sphere(sites, latitude)
            near = cKDTree(unit).query(self.centers.reshape(-1, 3))[1]
            anchors = unit[near].reshape(self.centers.shape)[..., None, :]
            self.ceiling = self.kernels(anchors)
            peak = self.kernels(self.centers[..., None, :])
            self.unresolved = -np.expm1(np.minimum(0., self.ceiling-peak))
            self.log_mass = (np.log(1e5)+draws["log_volume"][:, None]+np.log(draws["weights"]))[..., None]

    def kernels(self, unit):
        d = self.draws
        kw = d["kernel_weights"][:, None, :] if "kernel_weights" in d else None
        return log_kernel(unit, d["log_scale"], self.centers, d["exponent"][:, None],
                          d["tail_fraction"][:, None], d.get("exponential_fraction",
                          np.zeros(len(d["log_volume"])))[:, None], kw)

    def variants(self, xy, layout, states):
        unit = to_sphere(xy, self.latitude)
        if self.displaced:
            kernel = self.kernels(unit)
            excess = np.maximum(kernel-self.ceiling, 0.)
            parent = logsumexp(kernel+self.log_mass, axis=1)
        else:
            parent = log_field(self.draws, unit)
        freedom = layout.freedom(xy)
        across, beyond = layout.costs(xy, at_survey_edge=True)
        cost = np.min(across+beyond/4., axis=1)
        radius = np.linalg.norm(np.asarray(xy)*layout.scale,axis=1)
        outer = layout.lengths.max()
        leverage = 1./(1.+(np.maximum(radius-outer,0.)/outer)**2)
        aspect = layout.axial_concentration*leverage
        results = []
        for state in states:
            if state == "parent":
                results.append(parent)
                continue
            width, allowance = (1.5, np.log(4.)) if state == "broad" else (.75, np.log(2.))
            skirt = (.5 if state=="broad" else .25)*(1.-.75*aspect)
            log_spread = np.logaddexp(np.log(skirt), np.log1p(-skirt)+half_thickness(cost/width**2))
            if self.displaced:
                # Only the unsampled fraction of the central peak licenses extra lateral regularization.
                rise_removed = excess - allowance*np.tanh(excess/allowance)
                transverse = self.unresolved+(1.-self.unresolved)*aspect
                spread_weight = .5 if state=="broad" else 1.
                correction = freedom*(spread_weight*leverage*transverse*log_spread-rise_removed)
                results.append(logsumexp(kernel+self.log_mass+correction, axis=1))
            else:
                # Source-constrained kernels already impose an independently selected vent maximum.
                results.append(parent)
        return results
