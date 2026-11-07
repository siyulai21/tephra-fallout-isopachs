"""Whole-field posterior summaries; observation errors never add deposit mass."""
import numpy as np
from scipy.special import ndtr, logsumexp
from scipy.stats import qmc
from .geometry import to_sphere, from_sphere, centers, log_field, kernel_samples, RADIUS
from .inference import flatten, observation_log_likelihood


def quantiles(values):
    return dict(zip(("lower95", "median", "upper95"),
                    np.quantile(values, [.025, .5, .975], axis=0).tolist()))


def predictions(samples, data, latitude):
    """Integrate the predictive mixture CDF, including bounds and known errors."""
    flat = flatten(samples)
    mean = log_field(flat, to_sphere(data.xy, latitude))
    sigma = flat["noise"][:, None]
    sd = np.sqrt(sigma**2+data.log_sd[None, :]**2)
    broad_sd = np.sqrt(16*sigma**2+data.log_sd[None, :]**2)
    eps = flat["contamination"][:, None]
    result = {"log_score": logsumexp(observation_log_likelihood(flat, data, latitude), axis=0)-np.log(len(mean))}
    for name, probability in (("lower", .025), ("median", .5), ("upper", .975)):
        lower = np.min(mean-12*broad_sd, axis=0)
        upper = np.max(mean+12*broad_sd, axis=0)
        for _ in range(60):
            middle = (lower+upper)/2
            cdf = np.mean((1-eps)*ndtr((middle-mean)/sd)+eps*ndtr((middle-mean)/broad_sd), axis=0)
            lower = np.where(cdf < probability, middle, lower)
            upper = np.where(cdf >= probability, middle, upper)
        result[name] = (lower+upper)/2
    return result


def volume_summary(samples):
    flat = flatten(samples)
    values = np.exp(flat["log_volume"])
    return {**quantiles(values), "unit": "km3 bulk fallout", "draws": int(len(values)),
        "interpretation": "95% equal-tailed Bayesian credible interval conditional on this likelihood, prior and field family. Includes parameter and fan uncertainty; excludes unknown systematic preservation, sampling and source-attribution bias. Not dense-rock equivalent or ignimbrite volume."}


def mass_partition(samples, data, latitude, draws=1024, power=12):
    """Positive equal-mass quadrature of each fan, independent of any map box.

    Partition in priority order: local support footprint; unobserved proximal
    cap; beyond the most distant exact site; remaining gaps/sectors. These are
    sampling diagnostics, not geological deposit classes or confidence regions.
    """
    from ...support import SupportMesh
    mesh = SupportMesh(data)
    flat = flatten(samples)
    index = np.unique(np.linspace(0, len(flat["log_volume"])-1, min(draws, len(flat["log_volume"]))).astype(int))
    flat = {k: v[index] for k, v in flat.items()}
    center = centers(flat["log_scale"], flat["offset"])
    exact_unit = to_sphere(data.xy[~data.censored], latitude)
    radii = np.arccos(np.clip(exact_unit[:, 2], -1, 1))
    inner, outer = radii.min(), radii.max()
    uniforms = qmc.Sobol(3, scramble=True, seed=20261104).random_base2(power)
    fractions, fine_differences = [], []
    names = ("local_footprint", "unsampled_proximal", "beyond_farthest_site", "other_gaps_and_sectors")
    mass_errors = []
    for i in range(len(index)):
        row, coarse = np.zeros(4), np.zeros(4)
        if "core_log_scale" in flat:
            from .vent_envelope import gaussian_log_z, normalizer
            c = flat["fixed_center"][i] if "fixed_center" in flat else center[i]
            core = flat["core_log_scale"][i]
            strength = flat["envelope_strength"][i]
            norm = normalizer(core, flat["log_scale"][i], c, flat["weights"][i], strength)
            components = [(core, np.array([0.,0.,1.]), np.exp(gaussian_log_z(RADIUS**2*np.exp(-2*core))-norm), False)]
            for k, weight in enumerate(flat["weights"][i]):
                scale = flat["log_scale"][i,k]
                coefficient = strength*weight*np.exp(gaussian_log_z(RADIUS**2*np.exp(-2*scale))-norm)
                components.append((scale,c[k],coefficient,True))
            for scale,location,coefficient,shell in components:
                points = kernel_samples(scale,location,2.,0.,uniforms)
                xy = from_sphere(points,latitude)
                triangle = mesh.triangulation.find_simplex(xy)
                supported = (triangle >= 0) & mesh.accepted[np.maximum(triangle,0)]
                angle = np.arccos(np.clip(points[:,2],-1,1))
                part = np.where(supported,0,np.where(angle < inner,1,np.where(angle > outer,2,3)))
                importance = (-np.expm1(-.5*np.sum((points-[0.,0.,1.])**2,axis=1)*RADIUS**2*np.exp(-2*core))
                              if shell else np.ones(len(points)))
                row += coefficient*np.bincount(part,weights=importance,minlength=4)/len(part)
                half = len(part)//2
                coarse += coefficient*np.bincount(part[:half],weights=importance[:half],minlength=4)/half
            mass_errors.append(float(abs(row.sum()-1)))
            row /= row.sum(); coarse /= coarse.sum()
            fractions.append(row); fine_differences.append(abs(row-coarse))
            continue
        for k, weight in enumerate(flat["weights"][i]):
            if "angular" in flat:
                from .source_geometry import mass_samples
                points = mass_samples(flat["log_scale"][i, k], flat["angular"][i, k], flat["exponent"][i], flat["kernel_weights"][i], uniforms)
            else:
                points = kernel_samples(flat["log_scale"][i, k], center[i, k], flat["exponent"][i], flat["tail_fraction"][i], uniforms, flat.get("exponential_fraction", np.zeros(len(index)))[i])
            xy = from_sphere(points, latitude)
            triangle = mesh.triangulation.find_simplex(xy)
            supported = (triangle >= 0) & mesh.accepted[np.maximum(triangle, 0)]
            angle = np.arccos(np.clip(points[:, 2], -1, 1))
            part = np.where(supported, 0, np.where(angle < inner, 1, np.where(angle > outer, 2, 3)))
            row += weight*np.bincount(part, minlength=4)/len(part)
            coarse += weight*np.bincount(part[:len(part)//2], minlength=4)/(len(part)//2)
        fractions.append(row)
        fine_differences.append(abs(row-coarse))
    fractions = np.asarray(fractions)
    total = np.exp(flat["log_volume"])
    return {"parts": {name: {"fraction": quantiles(fractions[:, j]),
                    "volume_km3": quantiles(total*fractions[:, j])} for j, name in enumerate(names)},
        "n_posterior_fields": len(index), "quadrature_points_per_fan": len(uniforms),
        "max_fraction_change_on_doubling": float(np.max(fine_differences)),
        "max_partition_sum_error": float(abs(fractions.sum(axis=1)-1).max()),
        "max_positive_quadrature_mass_error_before_partition_normalization": max(mass_errors,default=0.),
        "nearest_exact_site_km": float(inner*RADIUS), "farthest_exact_site_km": float(outer*RADIUS),
        "interpretation": "Geometry-screened fractions, not fractions validated by data. Quantiles of parts do not add to quantiles of the total. Sobol integration error is separate from posterior uncertainty."}
