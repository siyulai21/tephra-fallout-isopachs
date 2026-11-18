"""Matched change-in-slope experiment for a withheld B-Tm locality.

Only the radial argument of the published reference likelihood changes. Distances,
fan geometry, local effects, bounds and observation noise keep their original units.
"""
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

from src.empirical_isopach.core.distributions import LogMixture
from src.empirical_isopach.models.directional import inference as base
from src.empirical_isopach.models.directional.geometry import BASE, Geometry
from src.empirical_isopach.models.directional.study import SEED, digest, fit, write_json

# Number of learned breaks, exponential-only kernels, and prior SD of each log slope ratio.
VARIANTS = {
    "smooth": (0, False, np.log(2.)), "one_break": (1, False, np.log(2.)),
    "two_breaks": (2, False, np.log(2.)), "exponential": (0, True, np.log(2.)),
    "broken_exponential": (1, True, np.log(2.)), "wide_prior": (1, False, np.log(4.)),
}


def warp(radius, breaks, ratios, xp=np):
    """Continuous piecewise-linear effective distance with positive, multiplicative slope changes."""
    radius = xp.asarray(radius)
    out = 0.
    start = 0.
    slope = 1.
    for i in range(breaks.shape[-1]):
        stop = breaks[..., i, None]
        out = out + slope*xp.maximum(0., xp.minimum(radius, stop)-start)
        start = stop
        slope = slope*xp.exp(ratios[..., i, None])
    return out + slope*xp.maximum(radius-start, 0.)


@lru_cache(maxsize=16)
def sampler(nbreaks, attempt, shapes):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model(r, vr, log_bounds, ratio_sd, **args):
        spacing = numpyro.sample("break_spacing", dist.Dirichlet(jnp.ones(nbreaks+1)))
        knots = jnp.exp(log_bounds[0] + jnp.cumsum(spacing)[:-1]*(log_bounds[1]-log_bounds[0]))
        ratios = numpyro.sample("log_slope_change", dist.Normal(0., ratio_sd).expand((nbreaks,)).to_event(1))
        base._model(r=warp(r, knots, ratios, jnp), vr=warp(vr, knots, ratios, jnp), **args)

    nuts = NUTS(model, target_accept_prob=.92 if attempt == 0 else .97, max_tree_depth=10)
    return MCMC(nuts, num_warmup=600*(attempt+1), num_samples=1000*2**attempt,
                num_chains=4, chain_method="vectorized", jit_model_args=True, progress_bar=False)


def curve(params, geometry, locations):
    radius, angular = geometry.coords(locations)
    if "break_km" in params:
        radius = warp(radius, params["break_km"], params["log_slope_change"])
    return base.background(params, radius, angular)


def sample_break(data, geometry, seed, attempt, nbreaks, ratio_sd, bounds):
    import jax

    args, relevant = base.training_arrays(geometry, data)
    kernel = sampler(nbreaks, attempt, tuple(np.shape(v) for v in args.values()))
    log_bounds = np.log(bounds)
    kernel.run(jax.random.PRNGKey(seed), **args, log_bounds=log_bounds, ratio_sd=ratio_sd,
               extra_fields=("energy","diverging","num_steps"))
    raw = {k:np.asarray(v) for k,v in kernel.get_samples(group_by_chain=True).items()}
    draws = {k:raw[k] for k in ("source","log_range","log_noise","fan_logit","log_slope_change")}
    draws["break_km"] = np.exp(log_bounds[0]+np.cumsum(raw["break_spacing"],axis=-1)[...,:-1]*np.diff(log_bounds)[0])
    draws["fan_log_range"] = raw["log_multiplier"][..., :len(geometry.centers)] + np.log(geometry.reaches)
    draws["tail_shape"] = (raw["tail_aux"] if geometry.policy.fixed_tail_shape is None
                           else np.zeros_like(raw["tail_aux"]))
    p = base.flat(draws)
    noise_var = np.exp(2*p["log_noise"])
    node = curve(p, geometry, geometry.vertices)
    var = np.broadcast_to((noise_var*geometry.policy.local_sd_ratio**2)[:,None],node.shape).copy()
    for i in np.flatnonzero(args["measured"][:len(data.xy)]):
        j = args["at"][i]
        gain = var[:,j]/(var[:,j]+noise_var+data.log_sd[i]**2)
        node[:,j] += gain*(data.log_value[i]-node[:,j])
        var[:,j] *= 1-gain
    z = np.random.default_rng(np.random.SeedSequence([seed,9743])).standard_normal(node.shape)
    z[:,relevant] = raw["innovation"].reshape(len(z),-1)[:,:len(relevant)]
    draws["nodes"] = (node+np.sqrt(var)*z).reshape((*draws["source"].shape,len(geometry.vertices)))
    extra = {k:np.asarray(v) for k,v in kernel.get_extra_fields(group_by_chain=True).items()}
    return draws, base.diagnostics(draws,extra)


def fit_variant(data, latitude, variant, fold, directory, reference=None):
    nbreaks, exponential, spread = VARIANTS[variant]
    policy = replace(BASE, name="exponential", fixed_tail_shape=0.) if exponential else BASE
    g = Geometry(data,latitude,policy)
    seed = SEED if fold is None else SEED+101+fold
    radius, _ = g.coords(data.xy)
    bounds = np.array([radius[radius>0].min(),radius.max()])
    if not nbreaks:
        target = reference if variant=="smooth" and reference is not None else directory
        draws, result = fit(data,g,seed,target)
        return g,draws,result,bounds
    signature = {"variant":variant,"seed":seed,"prior_sd":float(spread),"radius_limits_km":bounds.tolist(),
                 "geometry":g.describe(),"experiment_sha256":digest(Path(__file__)),
                 "reference_likelihood_sha256":digest(Path(base.__file__)),
                 "observations":{k:sha256(np.ascontiguousarray(getattr(data,k)).tobytes()).hexdigest()
                                 for k in ("xy","log_value","censored","lower_censored","row_ids","log_sd")}}
    directory.mkdir(exist_ok=True,parents=True)
    design = directory/"design.json"
    if design.exists() and json.loads(design.read_text()) != signature:
        raise ValueError(f"Changed inputs in {directory}")
    write_json(design, signature)
    history = []
    for attempt in range(2):
        info = directory/f"attempt-{attempt+1}.json"
        path = info.with_suffix(".npz")
        if info.exists():
            row=json.loads(info.read_text())
            if digest(path)!=row["posterior_sha256"]:
                raise ValueError(f"Changed posterior {path}")
            with np.load(path) as archive:
                draws=dict(archive)
        else:
            draws, diagnostics = sample_break(data,g,seed+1009*attempt,attempt,nbreaks,spread,bounds)
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, **draws)
            path.with_suffix(".tmp").replace(path)
            row={"attempt":attempt+1,"seed":seed+1009*attempt,"chains":4,"warmup":600*(attempt+1),
                 "draws_per_chain":1000*2**attempt,"diagnostics":diagnostics,"posterior_sha256":digest(path)}
            write_json(info,row)
        history.append(row)
        print(variant,fold,attempt+1,row["diagnostics"],flush=True)
        if row["diagnostics"]["passed"]:
            break
    return g,draws,{"attempts":history,"passed":history[-1]["diagnostics"]["passed"]},bounds


def prediction(draws, geometry, xy, log_sd=0.):
    p = base.flat(draws)
    _, ids, w = geometry.locate(xy)
    nodes = np.pad(p["nodes"],((0,0),(1,0)))
    means = np.maximum(0.,1-w.sum(axis=1))*curve(p,geometry,xy)+(nodes[:,ids+1]*w).sum(axis=-1)
    variances = np.exp(2*p["log_noise"])[:,None]*(1+geometry.policy.local_sd_ratio**2*np.maximum(0.,1-(w**2).sum(1)))+np.asarray(log_sd)**2
    return LogMixture(np.full(len(means),1/len(means)),means,variances)


def point_scores(dist, data):
    z = (data.log_value-dist.means)/np.sqrt(dist.variances)
    lp = np.where(data.censored,log_ndtr(np.where(data.lower_censored,-z,z)),
                  -.5*z*z-.5*np.log(2*np.pi*dist.variances))
    return logsumexp(lp,axis=0)-np.log(len(lp))
