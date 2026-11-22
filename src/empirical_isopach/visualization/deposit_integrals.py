"""Volume and superlevel area of the same median-thickness surface used to draw the maps."""
import numpy as np
from scipy.special import roots_legendre
from scipy.stats import qmc

from .deposit_field import PolarField


def within_cells(field, levels=(), order=4, angles=4):
    """Integrate the continuous bilinear log10 surface in equal-area polar coordinates.

    At angular quadrature nodes, all radial threshold crossings are linear in log thickness.
    Volume includes material thinner than the lowest drawn contour and all disconnected pieces.
    """
    r = field.radii
    z = field.log10_cm
    delta = np.roll(z, -1, axis=1) - z
    radii, weights = roots_legendre(order)
    offsets, bearings = roots_legendre(angles)
    volume = 0.
    areas = np.zeros(len(levels))
    dr = np.diff(r)[:, None]
    for offset, bw in zip((offsets+1)/2, bearings/2, strict=True):
        plane = z + offset*delta
        dz = np.diff(plane, axis=0)
        for pos, weight in zip((radii+1)/2, weights/2, strict=True):
            rho = r[:-1,None] + pos*dr
            thickness = 10.**(plane[:-1] + pos*dz)
            volume += bw*weight*np.sum(rho*dr*thickness)
        ray = PolarField(r, plane, field.axis)
        areas += bw*np.array([ray.area(cm) for cm in levels])
    return float(volume * (2*np.pi/z.shape[1]) * 1e-5), areas


def independent_integral(view, domain):
    """Two independent scrambled Sobol evaluations of the original saved field."""
    estimates = []
    for order in (12, 14, 16):
        trials = []
        halves = []
        for seed in (81173, 31991):
            u, v = qmc.Sobol(2, scramble=True, seed=seed).random_base2(order).T
            angle = 2*np.pi*u
            log_span = np.log1p(domain.rho_max/50.)
            radius = 50.*np.expm1(v*log_span)
            xy = np.column_stack((domain.axis_scale*radius*np.sin(angle),
                                  radius*np.cos(angle)/domain.axis_scale))
            weight = 2*np.pi*log_span*radius*(radius+50.)
            terms = 1e-5*weight*10.**view.central(xy)
            trials.append(float(terms.mean()))
            halves.append(float(terms[:len(terms)//2].mean()))
        mean = float(np.mean(trials))
        error = max(np.ptp(trials)/mean, abs(np.mean(halves)/mean-1.))
        estimates.append(dict(points_per_scramble=2**order, km3=trials, half_grid_km3=halves,
                              relative_change=float(error)))
        if error < .01:
            break
    return estimates
