"""Gaussian node effects with exact marginalization of repeated measurements.

NUTS samples the thinning curve, observation scale and the node innovations
needed by other observations (including bounds). Remaining innovations are
drawn from their exact Gaussian conditional distribution.
"""
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp, ndtr

from ...core.likelihood import normal_log_event, observation_log_event


def log_profile(radius, scale, shape):
    z = radius/scale
    return np.where(shape == 0., -z, -np.log1p(shape*z)/np.maximum(shape, 1e-300))


def background(params, radius, angular):
    core = np.exp(params["log_range"])[:, None]
    scale = core + np.exp(params["fan_log_range"]) @ angular.T
    mixing = params["fan_logit"][:, None]
    shape = params["tail_shape"][:, None]
    first = -np.logaddexp(0., mixing)+log_profile(radius, core, shape)
    second = -np.logaddexp(0., -mixing)+log_profile(radius, scale, shape)
    return params["source"][:, None] + np.logaddexp(first, second)


def field(params, geometry, xy):
    _, ids, w = geometry.locate(xy)
    mean = background(params, *geometry.coords(xy))
    nodes = np.pad(params["nodes"], ((0, 0), (1, 0)))
    return (1-w.sum(1)).clip(0, 1)*mean + (nodes[:, ids+1]*w).sum(axis=-1)


def flat(draws):
    return {k: v.reshape((v.shape[0]*v.shape[1], *v.shape[2:])) for k, v in draws.items()}


def predict(draws, geometry, data):
    samples = flat(draws)
    mu = field(samples, geometry, data.xy)
    _, _, weights = geometry.locate(data.xy)
    unresolved = 1+geometry.policy.local_sd_ratio**2*np.maximum(0., 1-(weights**2).sum(1))
    sd = np.hypot(np.exp(samples["log_noise"])[:, None]*np.sqrt(unresolved), data.log_sd)
    lp = observation_log_event(data, mu, sd)
    result = {"log_score": logsumexp(lp, axis=0)-np.log(len(mu))}
    for name, p in (("lower", .025), ("median", .5), ("upper", .975)):
        lo, hi = (mu-12*sd).min(0), (mu+12*sd).max(0)
        for _ in range(45):
            mid = (lo+hi)/2
            below = ndtr((mid-mu)/sd).mean(0) < p
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        result[name] = (lo+hi)/2
    return result


def _size(n, smallest):
    return max(smallest, 1 << max(0, int(n-1).bit_length()))


def _pad(a, shape, fill=0):
    out = np.full(shape, fill, dtype=a.dtype)
    out[tuple(slice(0, n) for n in a.shape)] = a
    return out


def training_arrays(g, data):
    n, v, k = len(data.xy), len(g.vertices), len(g.centers)
    vertices = {tuple(x): i for i, x in enumerate(g.vertices)}
    at = np.array([vertices.get(tuple(x), -1) for x in data.xy], int)
    measured = (at >= 0) & ~data.censored
    _, ids, w = g.locate(data.xy)
    w[measured] = 0.
    relevant = np.unique(ids[w > 0])
    r, angular = g.coords(data.xy)
    vr, va = g.coords(g.vertices)
    # Common padded shapes allow independent fits to reuse a compiled sampler.
    N, V, K, J = _size(n, 128), _size(v, 128), _size(k, 16), _size(len(relevant), 8)
    return dict(r=_pad(r, (N,)), ang=_pad(angular, (N, K)), vr=_pad(vr, (V,)),
        va=_pad(va, (V, K)), at=_pad(np.maximum(at, 0), (N,)), y=_pad(data.log_value, (N,)),
        error=_pad(data.log_sd, (N,)), measured=_pad(measured, (N,)),
        other=_pad(~measured, (N,)), cens=_pad(data.censored, (N,)),
        lower=_pad(data.lower_censored, (N,)), upper=_pad(data.interval_log_upper, (N,), np.nan),
        ids=_pad(np.maximum(ids, 0), (N, 3)),
        weights=_pad(w, (N, 3)), relevant=_pad(relevant, (J,), fill=V),
        fixed_shape=float(-1. if g.policy.fixed_tail_shape is None else g.policy.fixed_tail_shape), ratio=float(g.policy.local_sd_ratio),
        h0=float(g.policy.background_scale_km), reach=_pad(g.reaches, (K,))), relevant


def _model(r, ang, vr, va, at, y, error, measured, other, cens, lower, upper, ids,
           weights, relevant, fixed_shape, ratio, h0, reach):
    import jax.numpy as jnp
    from jax.scipy.special import log_ndtr as jlogcdf
    import numpyro
    from numpyro import distributions as dist

    source = numpyro.sample("source", dist.Normal(np.log(100.), 2.5))
    log_range = numpyro.sample("log_range", dist.Normal(jnp.log(h0), 1.4))
    fans = numpyro.sample("log_multiplier", dist.Normal(-np.log(3.), 1.).expand((ang.shape[1],)).to_event(1))
    mixing = numpyro.sample("fan_logit", dist.Normal(-1., 1.5))
    noise = jnp.exp(numpyro.sample("log_noise", dist.Normal(np.log(.7), .8)))
    innovations = numpyro.sample("innovation", dist.Normal(0., 1.).expand((len(relevant),)).to_event(1))
    candidate_shape = numpyro.sample("tail_aux", dist.Beta(.5, 1.5))
    shape = jnp.where(fixed_shape < 0., candidate_shape, fixed_shape)
    def radial(radius, scale):
        z = radius/scale
        return jnp.where(shape == 0., -z, -jnp.log1p(shape*z)/jnp.maximum(shape, 1e-30))
    def curve(radius, angles):
        core = jnp.exp(log_range)
        h = core + angles @ (reach*jnp.exp(fans))
        first = -jnp.logaddexp(0., mixing)+radial(radius, core)
        second = -jnp.logaddexp(0., -mixing)+radial(radius, h)
        return source + jnp.logaddexp(first, second)
    base = curve(vr, va)
    obs_var = noise**2+error**2
    tau2 = (noise*ratio)**2
    prec = measured/obs_var
    node_prec = jnp.zeros_like(vr).at[at].add(prec)
    var = tau2/(1+tau2*node_prec)
    shift = var*jnp.zeros_like(vr).at[at].add((y-base[at])*prec)
    node = base+shift
    joint = (jnp.sum(measured*jnp.log(2*np.pi*obs_var)+(y-node[at])**2*prec)
             +jnp.sum(shift**2/tau2+jnp.log1p(tau2*node_prec)))
    numpyro.factor("node_measurements", -.5*joint)
    # Padded indices equal len(vr) and are dropped by the scatter operation.
    node = node.at[relevant].add(jnp.sqrt(var.at[relevant].get(mode="clip"))*innovations, mode="drop")
    mu = (1-weights.sum(1))*curve(r, ang) + (weights*node[ids]).sum(1)
    # Missing local variance is unresolved site scatter. At a measured node
    # its variance is already carried by that node.
    obs_var = obs_var + tau2*jnp.maximum(0., 1-(weights**2).sum(1))
    log_prob = normal_log_event(y, mu, jnp.sqrt(obs_var), cens, lower, upper, xp=jnp, logcdf=jlogcdf)
    numpyro.factor("other_observations", jnp.sum(jnp.where(other, log_prob, 0.)))


@lru_cache(maxsize=12)
def _sampler(attempt, shapes):
    # NumPyro vectorizes the NUTS kernel in-place on initialisation. A change
    # of input shape needs its own kernel, including when jit_model_args is on.
    import jax
    jax.config.update("jax_enable_x64", True)
    from numpyro.infer import MCMC, NUTS
    sampler = NUTS(_model, target_accept_prob=.92 if attempt == 0 else .97, max_tree_depth=10)
    return MCMC(sampler, num_warmup=600*(attempt+1), num_samples=1000*2**attempt,
                num_chains=4, chain_method="vectorized", jit_model_args=True, progress_bar=False)


def sample(data, g, seed, attempt=0):
    import jax
    args, relevant = training_arrays(g, data)
    shapes = tuple(np.shape(a) for a in args.values())
    sampler = _sampler(attempt, shapes)
    sampler.run(jax.random.PRNGKey(seed), **args, extra_fields=("energy", "diverging", "num_steps"))
    raw = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
    draws = {k: raw[k] for k in ("source", "log_range", "log_noise", "fan_logit")}
    draws["fan_log_range"] = raw["log_multiplier"][..., :len(g.centers)] + np.log(g.reaches)
    draws["tail_shape"] = (raw["tail_aux"] if g.policy.fixed_tail_shape is None else
                           np.full_like(raw["tail_aux"], g.policy.fixed_tail_shape))
    p = flat(draws)
    s2 = np.exp(2*p["log_noise"])
    tau2 = s2*g.policy.local_sd_ratio**2
    node = background(p, *g.coords(g.vertices))
    var = np.broadcast_to(tau2[:, None], node.shape).copy()
    at = args["at"][:len(data.xy)]
    for i in np.flatnonzero(args["measured"][:len(data.xy)]):
        j = at[i]
        obs_var = s2+data.log_sd[i]**2
        gain = var[:, j]/(var[:, j]+obs_var)
        node[:, j] += gain*(data.log_value[i]-node[:, j])
        var[:, j] *= 1-gain
    z = np.random.default_rng(np.random.SeedSequence([seed, 9743])).standard_normal(node.shape)
    z[:, relevant] = raw["innovation"].reshape(len(z), -1)[:, :len(relevant)]
    draws["nodes"] = (node+np.sqrt(var)*z).reshape((*draws["source"].shape, len(g.vertices)))
    extra = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
    return draws, diagnostics(draws, extra)


def diagnostics(draws, extra):
    import arviz as az
    check = {k: v for k, v in draws.items() if v.size and np.any(np.ptp(v, axis=(0, 1)) > 1e-12)}
    dataset = az.from_dict(posterior=check)
    rh = max(float(v.max()) for v in az.rhat(dataset).data_vars.values())
    bulk = min(float(v.min()) for v in az.ess(dataset, method="bulk").data_vars.values())
    tail = min(float(v.min()) for v in az.ess(dataset, method="tail").data_vars.values())
    energy = extra["energy"]
    bfmi = (np.diff(energy, axis=1)**2).mean(1)/energy.var(1, ddof=1)
    div = int(extra["diverging"].sum())
    return {"rhat_max": rh, "bulk_ess_min": bulk, "tail_ess_min": tail, "divergences": div,
            "bfmi_min": float(bfmi.min()), "max_depth_fraction": float((extra["num_steps"] >= 1023).mean()),
            "passed": bool(rh < 1.01 and min(bulk, tail) >= 400 and div == 0 and bfmi.min() >= .3)}

