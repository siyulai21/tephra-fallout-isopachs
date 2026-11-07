"""Source-anchored directional fans on the complete authalic Earth.

Radial integrals are analytical; only a smooth periodic angular integral uses
quadrature. Every positive mixture is maximal at the source and nonincreasing
along each source-centred ray. No fitted patch centre or calculation box exists.
"""
import numpy as np
from scipy.special import logsumexp, gammainc, gammaincinv
from .geometry import RADIUS


def harmonics(angle, xp=np):
    return xp.stack((xp.cos(angle), xp.sin(angle), xp.cos(2*angle), xp.sin(2*angle)), axis=-1)


def elliptic_parameters(angular, xp=np):
    d2 = xp.sum(angular*angular, axis=-1)
    safe = xp.sqrt(xp.where(d2 > 1e-8, d2, 1.))
    cosh = xp.where(d2 > 1e-8, xp.cosh(safe), 1+d2/2+d2*d2/24)
    sinhc = xp.where(d2 > 1e-8, xp.sinh(safe)/safe, 1+d2/6+d2*d2/120)
    return cosh, sinhc


def directional_scales(log_scale, angular, angle, xp=np):
    if angular.shape[-1] == 4:
        return log_scale[..., None]+angular@harmonics(angle, xp).T
    cosh, sinhc = elliptic_parameters(angular, xp)
    direction = xp.stack((xp.cos(angle), xp.sin(angle)), axis=-1)
    denominator = cosh[..., None]-sinhc[..., None]*(angular@direction.T)
    # sqrt(cosh) makes L the planar area-equivalent radial scale.
    return log_scale[..., None]-.5*xp.log(cosh)[..., None]-xp.log(denominator)


def radial_normalizers(log_scale, q, xp=np):
    """Log of full-circle radial integrals, broadcast over angle."""
    ls = log_scale
    l2 = xp.exp(2*ls)/RADIUS**2
    zg = xp.log(2*xp.pi*RADIUS**2)+2*ls-2*xp.log(RADIUS)+xp.log(-xp.expm1(-2/l2))
    b = q*l2
    h = xp.log1p(2/b)
    t = (1-q)*h
    small = abs(t) < 1e-4
    safe = xp.where(small, 1., t)
    regular = xp.log(abs(xp.expm1(xp.minimum(safe, 50))))-xp.log(abs(safe))
    large = safe+xp.log1p(-xp.exp(-xp.maximum(safe, 50)))-xp.log(abs(safe))
    exprel = xp.where(small, t/2+t*t/24-t**4/2880, xp.where(safe > 50, large, regular))
    zp = xp.log(2*xp.pi*RADIUS**2)+xp.log(b)+xp.log(h)+exprel
    x = 2*RADIUS*xp.exp(-ls)
    small = x < 1e-3
    safe = xp.where(small, 1., x)
    regular = xp.log(-xp.expm1(xp.log1p(safe)-safe))
    t = xp.where(small, x, 0.)
    series = 2*xp.log(x)-xp.log(2)+xp.log1p(-2*t/3+t*t/4-t**3/15+t**4/72-t**5/420)
    ze = xp.log(2*xp.pi)+2*ls+xp.where(small, series, regular)
    return xp.stack((zg, ze, zp), axis=-2)


def normalizers(log_scale, angular, q, nodes=256, xp=np, lse=logsumexp):
    angle = xp.arange(nodes)*2*xp.pi/nodes
    if angular.shape[-1] == 4:
        scales = directional_scales(log_scale, angular, angle, xp)
        jacobian = 0.
    else:
        # Eccentric anomaly resolves a narrow downwind fan without needing an
        # angular grid concentrated by trial and error around its fitted axis.
        cosh, sinhc = elliptic_parameters(angular, xp)
        d2 = xp.sum(angular*angular, axis=-1)
        delta = xp.sqrt(xp.where(d2 > 0, d2, 1.))
        e = xp.where(d2 > 0, xp.tanh(delta), 0.)
        factor = 1+e[..., None]*xp.cos(angle)
        scales = log_scale[..., None]+.5*xp.log(cosh)[..., None]+xp.log(factor)
        jacobian = -xp.log(cosh)[..., None]-xp.log(factor)
    radial = radial_normalizers(scales, q[..., None], xp)
    if angular.shape[-1] == 2:
        radial = radial+jacobian[..., None, :]
    return lse(radial, axis=-1)-xp.log(float(nodes))


def log_kernel(unit, log_scale, angular, q, kernel_weights, nodes=256, xp=np, lse=logsumexp):
    angle = xp.arctan2(unit[:, 1], unit[:, 0])
    scales = directional_scales(log_scale, angular, angle, xp)
    # Chord radius is the equal-area polar radius: dA = r dr dtheta.
    u = .5*xp.sum((unit-xp.array([0., 0., 1.]))**2, axis=-1)
    r = xp.where(u > 0, xp.sqrt(2*xp.where(u > 0, u, 1.)), 0.)*RADIUS
    z = r*xp.exp(-scales)
    raw = xp.stack((-.5*z*z, -z, -q[..., None]*xp.log1p(z*z/(2*q[..., None]))), axis=-2)
    norm = normalizers(log_scale, angular, q, nodes, xp, lse)
    return lse(raw-norm[..., None]+xp.log(kernel_weights)[..., :, None], axis=-2)


def mass_samples(log_scale, angular, q, kernel_weights, uniforms, angular_nodes=4096):
    """Positive quadrature samples via angular CDF and exact radial CDFs.

    Angular CDF uses a periodic trapezoid, separately for each kernel. This
    numerical partition diagnostic does not determine the total volume.
    """
    angle = np.linspace(0, 2*np.pi, angular_nodes+1)
    elliptic = len(angular) == 2
    if elliptic:
        delta = np.linalg.norm(angular)
        cosh, e = np.cosh(delta), np.tanh(delta)
        factor = 1+e*np.cos(angle)
        scales = log_scale+.5*np.log(cosh)+np.log(factor)
        log_z = radial_normalizers(scales, q)-np.log(cosh)-np.log(factor)
    else:
        scales = directional_scales(np.asarray(log_scale), angular, angle)
        log_z = radial_normalizers(scales, q)
    cdf = np.cumsum(kernel_weights)
    choice = np.minimum(np.searchsorted(cdf, uniforms[:, 0]), 2)
    result = np.empty((len(uniforms), 3))
    for j in range(3):
        selected = choice == j
        if not selected.any():
            continue
        density = np.exp(log_z[j]-np.max(log_z[j]))
        angular_cdf = np.r_[0., np.cumsum((density[:-1]+density[1:])/2)]
        angular_cdf /= angular_cdf[-1]
        theta = np.interp(uniforms[selected, 2], angular_cdf, angle)
        if elliptic:
            ls = log_scale+.5*np.log(cosh)+np.log1p(e*np.cos(theta))
            theta = np.arctan2(np.sin(theta)/cosh, e+np.cos(theta))+np.arctan2(angular[1], angular[0])
        else:
            ls = directional_scales(np.asarray(log_scale), angular, theta)
        l2 = np.exp(2*ls)/RADIUS**2
        probability = uniforms[selected, 1]
        if j == 0:
            u = -l2*np.log1p(probability*np.expm1(-2/l2))
        elif j == 1:
            radial = np.exp(ls)*gammaincinv(2., probability*gammainc(2., 2*RADIUS*np.exp(-ls)))
            u = .5*(radial/RADIUS)**2
        else:
            b, a = q*l2, 1-q
            h = np.log1p(2/b)
            u = b*np.expm1(probability*h) if abs(a) < 1e-8 else b*np.expm1(np.log1p(probability*np.expm1(a*h))/a)
        u = np.clip(u, 0., 2.)
        transverse = np.sqrt(u*(2-u))
        result[selected] = np.column_stack((transverse*np.cos(theta), transverse*np.sin(theta), 1-u))
    return result
