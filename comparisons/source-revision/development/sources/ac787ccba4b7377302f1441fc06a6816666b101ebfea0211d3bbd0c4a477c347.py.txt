"""NUTS inference with exact amplitude centering and label-invariant checks."""
from dataclasses import dataclass, asdict
from functools import lru_cache
import time

import numpy as np
from scipy.special import logsumexp, log_ndtr

from .geometry import RADIUS, to_sphere, log_field
from .prior import BASELINE


@dataclass(frozen=True)
class Sampling:
    warmup: int = 1000
    draws: int = 1500
    chains: int = 4
    target_accept: float = .95
    max_tree_depth: int = 11
    seed: int = 20261104

    def __post_init__(self):
        if self.warmup < 1 or self.draws < 20 or self.chains < 2:
            raise ValueError("Require warmup, at least 20 draws, and independent multiple chains.")
        if not 0 < self.target_accept < 1 or self.max_tree_depth < 2:
            raise ValueError("Invalid NUTS settings.")


@lru_cache(maxsize=12)
def model_function(prior):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.scipy as jsp
    import numpyro
    import numpyro.distributions as dist

    def exprel(t):
        small = abs(t) < 1e-4
        s = jnp.where(small, 1., t)
        # Here t=(1-q)*log(1+2/b); positive t is bounded by log(1+2/b).
        value = jnp.log(abs(jnp.expm1(jnp.minimum(s, 50))))-jnp.log(abs(s))
        large = s+jnp.log1p(-jnp.exp(-jnp.maximum(s, 50)))-jnp.log(abs(s))
        return jnp.where(small, t/2+t*t/24-t**4/2880, jnp.where(s > 50, large, value))

    def model(unit, y, known_sd, kinds, mask, basis_centers=None, basis_scales=None):
        k = basis_centers.shape[0] if prior.supported_basis else prior.components
        # This is an exact change of coordinates, not a thickness-derived prior.
        # The factor below cancels this auxiliary normal and restores p(log V).
        level = numpyro.sample("field_level", dist.Normal(0., prior.log_volume_sd))
        if prior.supported_basis:
            width = numpyro.sample("basis_width", dist.Normal(prior.basis_width_log_mean, prior.basis_width_log_sd))
            ls = numpyro.deterministic("log_scale", jnp.log(basis_scales)+width)
            fixed_center = numpyro.deterministic("fixed_center", basis_centers)
            offset = numpyro.deterministic("offset", jnp.zeros((k, 2)))
        else:
            ls = numpyro.sample("log_scale", dist.Normal(prior.log_scale_mean, prior.log_scale_sd).expand((k,)).to_event(1))
        if prior.supported_basis:
            pass
        elif prior.source_anchored:
            dimension = 2 if prior.source_shape == "elliptic" else 4
            angular = numpyro.sample("angular", dist.Normal(0., prior.angular_sd).expand((k, dimension)).to_event(2))
            numpyro.deterministic("angular_nodes", jnp.array(prior.angular_nodes))
            offset = numpyro.deterministic("offset", jnp.zeros((k, 2)))
        else:
            offset = numpyro.sample("offset", dist.Normal(0., prior.offset_sd).expand((k, 2)).to_event(2))
        weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(k)*prior.fan_concentration/k)) if k > 1 else numpyro.deterministic("weights", jnp.ones(1))
        if prior.vent_envelope:
            core_log_scale = numpyro.sample("core_log_scale", dist.Normal(prior.log_scale_mean, prior.log_scale_sd))
            envelope_strength = numpyro.sample("envelope_strength", dist.Beta(1., 1.))
            kernel_weights = numpyro.deterministic("kernel_weights", jnp.array([1., 0., 0.]))
        elif prior.exponential_core:
            kernel_weights = numpyro.sample("kernel_weights", dist.Dirichlet(jnp.ones(3)*prior.kernel_concentration/3))
        else:
            two = numpyro.sample("two_kernel_weights", dist.Dirichlet(jnp.ones(2)*prior.kernel_concentration/2))
            kernel_weights = numpyro.deterministic("kernel_weights", jnp.array([two[0], 0., two[1]]))
        numpyro.deterministic("tail_fraction", kernel_weights[2])
        numpyro.deterministic("exponential_fraction", kernel_weights[1]/(kernel_weights[0]+kernel_weights[1]))
        if prior.vent_envelope:
            q = numpyro.deterministic("exponent", 2.)
        else:
            h = numpyro.sample("tail_log", dist.Normal(prior.tail_log_mean, prior.tail_log_sd))
            q = numpyro.deterministic("exponent", (1. if prior.integrable_plane_tail else 0.)+jnp.exp(h))
        sigma = numpyro.sample("noise", dist.HalfNormal(prior.noise_scale))
        epsilon = numpyro.sample("contamination", dist.Beta(1., 9.)) if prior.contamination else numpyro.deterministic("contamination", 0.)
        if prior.vent_envelope:
            from .vent_envelope import log_shape as envelope_shape
            log_shape = envelope_shape(unit, core_log_scale, ls, offset, weights, envelope_strength,
                                       jnp, jsp.special.logsumexp,
                                       center=fixed_center if prior.supported_basis else None)
        else:
            if prior.source_anchored:
                from .source_geometry import log_kernel as source_kernel
                kernel = source_kernel(unit, ls, angular, q*jnp.ones(k), kernel_weights,
                                       prior.angular_nodes, jnp, jsp.special.logsumexp)
            else:
                a = jnp.exp(ls)[:, None]*offset/(2*RADIUS)
                a2 = jnp.sum(a*a, axis=-1)
                center = jnp.concatenate((2*a, (1-a2)[:, None]), axis=1)/(1+a2)[:, None]
                u = .5*jnp.sum((center[:, None, :]-unit[None, :, :])**2, axis=-1)
                l2 = jnp.exp(2*ls)/RADIUS**2
                zg = jnp.log(2*jnp.pi*RADIUS**2)+jnp.log(l2)+jnp.log(-jnp.expm1(-2/l2))
                b = q*l2
                hh = jnp.log1p(2/b)
                zp = jnp.log(2*jnp.pi*RADIUS**2)+jnp.log(b)+jnp.log(hh)+exprel((1-q)*hh)
                gaussian = -u/l2[:, None]-zg[:, None]
                if prior.exponential_core:
                    x = 2*RADIUS*jnp.exp(-ls)
                    small = x < 1e-3
                    safe = jnp.where(small, 1., x)
                    regular = jnp.log(-jnp.expm1(jnp.log1p(safe)-safe))
                    t = jnp.where(small, x, 0.)
                    series = 2*jnp.log(x)-jnp.log(2)+jnp.log1p(-2*t/3+t*t/4-t**3/15+t**4/72-t**5/420)
                    ze = jnp.log(2*jnp.pi)+2*ls+jnp.where(small, series, regular)
                    # Define a zero gradient at the measure-zero centre, avoiding 0/0
                    # while retaining the exact exponential density value there.
                    radial = jnp.where(u > 0, jnp.sqrt(2*jnp.where(u > 0, u, 1.)), 0.)
                    exponential = -radial/jnp.sqrt(l2)[:, None]-ze[:, None]
                heavy = -q*jnp.log1p(u/b[:, None])-zp[:, None]
                if prior.exponential_core:
                    kernel = jsp.special.logsumexp(jnp.stack((gaussian, exponential, heavy), axis=0)
                                                   +jnp.log(kernel_weights)[:, None, None], axis=0)
                else:
                    kernel = jnp.logaddexp(jnp.log(kernel_weights[0])+gaussian, jnp.log(kernel_weights[2])+heavy)
            log_shape = jnp.log(1e5)+jsp.special.logsumexp(jnp.log(weights)[:, None]+kernel, axis=0)
        centering = jnp.sum(jnp.where(mask, log_shape, 0.))/jnp.sum(mask)
        lv = numpyro.deterministic("log_volume", level-centering)
        numpyro.factor("volume_prior", dist.Normal(prior.log_volume_mean, prior.log_volume_sd).log_prob(lv)
                       -dist.Normal(0., prior.log_volume_sd).log_prob(level))
        mean = level+log_shape-centering

        def likelihood(multiplier):
            sd = jnp.sqrt((multiplier*sigma)**2+known_sd**2)
            z = (y-mean)/sd
            exact = -.5*(z*z+jnp.log(2*jnp.pi))-jnp.log(sd)
            return jnp.where(kinds == 1, jsp.special.log_ndtr(z),
                             jnp.where(kinds == 2, jsp.special.log_ndtr(-z), exact))
        value = likelihood(1.)
        if prior.contamination:
            value = jnp.logaddexp(jnp.log1p(-epsilon)+value, jnp.log(epsilon)+likelihood(4.))
        numpyro.factor("observations", jnp.sum(jnp.where(mask, value, 0.)))
    return model


def packed_data(data, latitude):
    n = len(data.xy)
    size = max(128, 2**int(np.ceil(np.log2(n))))
    unit = np.tile([0., 0., 1.], (size, 1)); unit[:n] = to_sphere(data.xy, latitude)
    y = np.zeros(size); y[:n] = data.log_value
    sd = np.zeros(size); sd[:n] = data.log_sd
    kind = np.zeros(size, int); kind[:n] = np.where(data.lower_censored, 2, np.where(data.censored, 1, 0))
    return unit, y, sd, kind, np.arange(size) < n


def flatten(samples):
    return {key: np.asarray(value).reshape((-1,)+np.asarray(value).shape[2:]) for key, value in samples.items()}


def diagnostic_summary(samples, extras, data, latitude):
    import arviz as az
    flat = flatten(samples)
    n_chains, n_draws = samples["log_volume"].shape[:2]
    # Summaries are invariant to permutation of exchangeable fan labels.
    check = {k: samples[k] for k in ("log_volume", "noise", "tail_fraction", "exponent", "exponential_fraction")}
    w, ls = samples["weights"], samples["log_scale"]
    check["weighted_log_scale"] = np.sum(w*ls, axis=-1)
    check["effective_fans"] = 1/np.sum(w*w, axis=-1)
    if "core_log_scale" in samples:
        check["core_log_scale"] = samples["core_log_scale"]
        check["envelope_strength"] = samples["envelope_strength"]
    if "basis_width" in samples:
        check["basis_width"] = samples["basis_width"]
        # Fixed locations identify their labels: weight permutations are not
        # equivalent fields. Check each weight as well as aggregate summaries.
        for j in range(w.shape[-1]):
            check[f"basis_weight_{j}"] = w[..., j]
    if "angular" in samples:
        for j in range(samples["angular"].shape[-1]):
            check[f"weighted_angular_{j}"] = np.sum(w*samples["angular"][..., j], axis=-1)
    index = np.linspace(0, len(data.xy)-1, min(16, len(data.xy))).astype(int)
    fields = log_field(flat, to_sphere(data.xy[index], latitude)).reshape(n_chains, n_draws, -1)
    for j in range(fields.shape[-1]):
        check[f"field_{j}"] = fields[:, :, j]
    # Probe the source and distant unsampled sectors as well as fitted sites.
    angles = np.arange(4)*np.pi/2
    theta = np.array([100., 1000., 10000.])/RADIUS
    probes = np.array([[np.sin(t)*np.cos(a), np.sin(t)*np.sin(a), np.cos(t)]
                       for t in theta for a in angles]+[[0., 0., 1.]])
    fields = log_field(flat, probes).reshape(n_chains, n_draws, -1)
    for j in range(fields.shape[-1]):
        check[f"extrapolation_{j}"] = fields[:, :, j]
    # K=1 has an exactly constant effective fan count, not a failed chain.
    check = {k: v for k, v in check.items() if np.ptp(v) > 1e-12}
    idata = az.from_dict(posterior=check)
    rh = az.rhat(idata, method="rank").to_array().values
    ess = az.ess(idata, method="bulk").to_array().values
    tail = az.ess(idata, method="tail").to_array().values
    per_volume = {"rhat": float(az.rhat(idata, var_names=["log_volume"], method="rank").log_volume),
                  "bulk_ess": float(az.ess(idata, var_names=["log_volume"], method="bulk").log_volume),
                  "tail_ess": float(az.ess(idata, var_names=["log_volume"], method="tail").log_volume)}
    per_volume["log_quantile_mcse"] = {str(p): float(az.mcse(idata, var_names=["log_volume"], method="quantile", prob=p).log_volume)
                                      for p in (.025, .5, .975)}
    divergence = int(np.sum(extras["diverging"]))
    energy = np.asarray(extras["energy"])
    bfmi = np.mean(np.diff(energy, axis=1)**2, axis=1)/np.var(energy, axis=1, ddof=1)
    return {"volume": per_volume, "max_label_invariant_rhat": float(np.max(rh)),
        "min_label_invariant_bulk_ess": float(np.min(ess)), "min_label_invariant_tail_ess": float(np.min(tail)),
        "divergences": divergence, "energy_bfmi_per_chain": bfmi.tolist(), "min_energy_bfmi": float(bfmi.min()),
        "label_invariant_variables": list(check),
        "chain_volume_quantiles_km3": [np.exp(np.quantile(s, [.025, .5, .975])).tolist() for s in samples["log_volume"]],
        "passed": bool(np.max(rh) < 1.01 and np.min(ess) >= 400 and np.min(tail) >= 400 and divergence == 0 and bfmi.min() >= .3),
        "interpretation": "Rank-normalized split R-hat and bulk/tail ESS across independent chains. Fan labels are exchangeable; field and volume diagnostics are used instead. Passing diagnostics does not prove global mode exploration or model correctness."}


def sample(data, latitude, prior=BASELINE, settings=Sampling(), progress=print):
    import jax
    jax.config.update("jax_enable_x64", True)
    from numpyro.infer import MCMC, NUTS, init_to_median
    data = data.canonical()
    start = time.monotonic()
    nuts = NUTS(model_function(prior), target_accept_prob=settings.target_accept,
                max_tree_depth=settings.max_tree_depth, init_strategy=init_to_median(num_samples=20))
    mcmc = MCMC(nuts, num_warmup=settings.warmup, num_samples=settings.draws,
                num_chains=settings.chains, chain_method="sequential", progress_bar=False, jit_model_args=True)
    design = {}
    if prior.supported_basis:
        from .supported_basis import design as basis_design
        center, scales = basis_design(data, latitude, prior.components)
        design = {"basis_centers": center, "basis_scales": scales}
    mcmc.run(jax.random.PRNGKey(settings.seed), *packed_data(data, latitude), **design, extra_fields=("diverging", "num_steps", "accept_prob", "energy"))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    extras = {k: np.asarray(v) for k, v in mcmc.get_extra_fields(group_by_chain=True).items()}
    diagnostics = diagnostic_summary(samples, extras, data, latitude)
    diagnostics.update(settings=asdict(settings), elapsed_seconds=time.monotonic()-start,
        max_depth_fraction=float(np.mean(extras["num_steps"] >= 2**settings.max_tree_depth-1)),
        mean_acceptance=float(np.mean(extras["accept_prob"])))
    if progress:
        progress(f"{prior.name}: volume median {np.exp(np.median(samples['log_volume'])):.2f} km³; R-hat {diagnostics['max_label_invariant_rhat']:.3f}; divergences {diagnostics['divergences']}; {diagnostics['elapsed_seconds']:.1f} s")
    return samples, diagnostics


def observation_log_likelihood(samples, data, latitude):
    field = log_field(samples, to_sphere(data.xy, latitude))
    parts = []
    for mult in (1., 4.):
        sd = np.sqrt((mult*samples["noise"][:, None])**2+data.log_sd[None, :]**2)
        z = (data.log_value-field)/sd
        value = -.5*(z*z+np.log(2*np.pi))-np.log(sd)
        value[:, data.censored] = log_ndtr(z[:, data.censored])
        value[:, data.lower_censored] = log_ndtr(-z[:, data.lower_censored])
        parts.append(value)
    eps = samples["contamination"][:, None]
    with np.errstate(divide="ignore"):
        return np.logaddexp(np.log1p(-eps)+parts[0], np.log(eps)+parts[1])
