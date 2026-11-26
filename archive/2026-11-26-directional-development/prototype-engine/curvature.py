"""A scale-relative bending prior on the fitted field, never on extracted lines."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

from .geometry import WARP_KM
from ...mass_mixture.geometry import RADIUS


def covariance(a, b, length):
    """Positive two-scale Gaussian convolution covariance in the equal-area chart."""
    aa = WARP_KM**2 + np.sum(a*a, axis=1)
    bb = WARP_KM**2 + np.sum(b*b, axis=1)
    total = aa[:, None] + bb[None, :]
    distance = cdist(a, b, "sqeuclidean") / (length**2 * total)
    gain = 2*np.sqrt(aa[:, None]*bb[None, :])/total
    return gain*(.8*np.exp(-distance) + .2*np.exp(-9*distance))


def hessians(a, b, length, kernel="gaussian-mixture", *, with_gradient=False):
    """Analytic xx, xy and yy derivatives, including the antipodal rim factor."""
    aa = WARP_KM**2 + np.sum(a*a, axis=1)
    bb = WARP_KM**2 + np.sum(b*b, axis=1)
    total = aa[:, None] + bb[None, :]
    x, y = b.T
    dx, dy = x[None, :]-a[:, 0, None], y[None, :]-a[:, 1, None]
    squared = dx*dx+dy*dy
    gain = 2*np.sqrt(aa[:, None]*bb[None, :])/total
    ux, uy = x/bb-2*x/total, y/bb-2*y/total
    uxx = 1/bb-2*x*x/bb**2-2/total+4*x*x/total**2
    uyy = 1/bb-2*y*y/bb**2-2/total+4*y*y/total**2
    uxy = -2*x*y/bb**2+4*x*y/total**2
    qx, qy = 2*dx/total-2*squared*x/total**2, 2*dy/total-2*squared*y/total**2
    qxx = 2/total-8*dx*x/total**2-2*squared/total**2+8*squared*x*x/total**3
    qyy = 2/total-8*dy*y/total**2-2*squared/total**2+8*squared*y*y/total**3
    qxy = -4*(dx*y+x*dy)/total**2+8*squared*x*y/total**3
    rim = 1-np.sum(b*b, axis=1)/(2*RADIUS)**2
    rx, ry = -2*b.T/(2*RADIUS)**2
    rxx = -2/(2*RADIUS)**2
    result = np.zeros((3, len(a), len(b)))
    gradient = np.zeros((2, len(a), len(b)))
    for weight, ell in ((.8, length), (.2, length/3)):
        if kernel == "matern-v5":
            k = 10/ell**2
            d = np.sqrt(np.maximum(0., k*squared/total))
            ex = np.exp(-d)
            f = (1+d+d*d/3)*ex
            fp = -k/6*(1+d)*ex
            fpp = k*k/12*ex
        else:
            k = 1/ell**2
            f = np.exp(-k*squared/total)
            fp, fpp = -k*f, k*k*f
        value = weight*gain*f
        gx = weight*gain*(ux*f+fp*qx)
        gy = weight*gain*(uy*f+fp*qy)
        xx = weight*gain*((uxx+ux*ux)*f+2*ux*fp*qx+fpp*qx*qx+fp*qxx)
        yy = weight*gain*((uyy+uy*uy)*f+2*uy*fp*qy+fpp*qy*qy+fp*qyy)
        xy = weight*gain*((uxy+ux*uy)*f+fp*(ux*qy+uy*qx)+fpp*qx*qy+fp*qxy)
        gradient[0] += rim*gx+rx*value
        gradient[1] += rim*gy+ry*value
        result[0] += rim*xx+2*rx*gx+rxx*value
        result[1] += rim*xy+rx*gy+ry*gx
        result[2] += rim*yy+2*ry*gy+rxx*value
    return (result, gradient) if with_gradient else result


def energy(base, order, tension_ratio=0.):
    """Uniform angle and log-radius integration of the squared scaled Hessian.

    Align only the numerical rule with the training coordinates. Rotating an
    input rotates the probes too; it supplies no preferred dispersal direction.
    """
    node, weight = leggauss(order)
    maximum = np.log(np.hypot(WARP_KM, 2*RADIUS)/WARP_KM)
    radii = WARP_KM*np.sqrt(np.expm1(maximum*(node+1)))
    center = base.centers.mean(axis=0)
    phase = np.arctan2(center[0], center[1]) if np.linalg.norm(center)>1e-10 else 0.
    angle = phase+np.arange(2*order)*np.pi/order
    points = np.stack((np.outer(radii, np.sin(angle)), np.outer(radii, np.cos(angle))), axis=-1).reshape(-1, 2)
    weights = np.repeat(weight/(4*order), 2*order)
    result = np.zeros((len(base.averaging), len(base.averaging)))
    for first in range(0, len(points), 256):
        block = slice(first, first+256)
        point = points[block]
        derivative, gradient = hessians(base.centers, point, base.length, base.kernel, with_gradient=True)
        scale = WARP_KM**2+np.sum(point*point, axis=1)
        for h, multiplier in zip(derivative, (1., 2., 1.), strict=True):
            projected = (base.averaging@h)*scale
            result += multiplier*(projected*weights[block])@projected.T
        if tension_ratio:
            for g in gradient:
                projected = (base.averaging@g)*np.sqrt(scale)
                result += tension_ratio*(projected*weights[block])@projected.T
    return (result+result.T)/2


def regularize(base, strength=.05, *, tolerance=.03, maximum_order=384, amplitude_precision=.2, tension=False):
    """Whiten a proper Gaussian bending prior and retain its quadrature checks."""
    from dataclasses import replace

    n = len(base.averaging)
    if not n:
        return replace(base, reference=base.averaging,
                       diagnostics=dict(strength=strength, passed=True, refinement=[]))
    previous = None
    history = []
    for order in (24, 48, 96, 192, 384):
        if order>maximum_order:break
        precision = amplitude_precision*np.eye(n)+strength*energy(base, order, 2*np.sqrt(amplitude_precision/strength) if tension else 0.)
        factor = np.linalg.cholesky(precision)
        change = None
        if previous is not None:
            a = solve_triangular(factor, precision-previous, lower=True, check_finite=False)
            relative = solve_triangular(factor, a.T, lower=True, check_finite=False).T
            change = float(np.max(abs(np.linalg.eigvalsh((relative+relative.T)/2))))
        passed = change is not None and change<tolerance
        history.append(dict(radial_nodes=order, angular_nodes=2*order,
                            maximum_relative_precision_change=change, passed=passed))
        if passed:break
        previous = precision
    return replace(base, averaging=solve_triangular(factor, base.averaging, lower=True, check_finite=False),
                   reference=base.averaging,
                   diagnostics=dict(strength=strength, amplitude_precision=amplitude_precision, tension=tension,
                                    relative_tolerance=tolerance, passed=passed, refinement=history))
