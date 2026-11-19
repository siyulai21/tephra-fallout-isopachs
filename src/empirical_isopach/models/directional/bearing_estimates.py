"""Whole-field and median-map integration for support-aware directional continuations."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .bearing_support import complete_masses
from .ensemble.fields import Ensemble, Member, draw_weights, inverse_cdf, latent_arrays, subset_draws
from .ensemble.integrals import ORDERS, SEEDS, interval, polar_nodes
from .geometry import polar
from .study import digest, write_json


def name(member):
    return f"{member.geometry.policy.name}-{member.geometry.ratio:g}"


def mass_draws(member, folder):
    import arviz as az
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / (name(member) + ".json")
    path = folder / (name(member) + ".npz")
    signature = {"posterior_sha256": member.records[-1]["posterior_sha256"],
                 "geometry": member.geometry.describe(),
                 "integration_sources": {p:digest(Path(__file__).with_name(p)) for p in
                                         ("integration.py", "bearing_support.py", "bearing_estimates.py")}}
    if meta.is_file():
        record = json.loads(meta.read_text())
        if record["signature"] == signature and path.is_file() and record["sha256"] == digest(path):
            with np.load(path) as archive:
                return archive["mass"], record
    mass, convergence = complete_masses(member.draws, member.geometry)
    total = mass.sum(axis=(-1,-2))
    chains = az.from_dict(posterior={"log_volume": np.log(total)})
    rh = float(az.rhat(chains).log_volume)
    bulk = float(az.ess(chains, method="bulk").log_volume)
    tail = float(az.ess(chains, method="tail").log_volume)
    volume_check = {"rhat":rh,"bulk_ess":bulk,"tail_ess":tail,"passed":rh < 1.01 and min(bulk,tail) >= 400}
    with path.with_suffix(".tmp").open("wb") as out:
        np.savez_compressed(out, mass=mass)
    path.with_suffix(".tmp").replace(path)
    record = {"signature":signature,"sha256":digest(path),"quadrature":convergence,
              "volume_diagnostics":volume_check,"sampling_diagnostics":member.records[-1]["diagnostics"],
              "passed":bool(convergence["passed"] and volume_check["passed"] and member.records[-1]["diagnostics"]["passed"])}
    write_json(meta, record)
    return mass, record


def packet(model, count):
    members = []
    for member in model.members:
        draws, indices = subset_draws(member.draws, count)
        members.append(Member(member.geometry, draws, indices, member.records))
    return Ensemble(members,model.model_weights)


def integrate_points(model, uv, levels):
    geometry = model.members[0].geometry
    radii = polar(geometry.mesh.xy,geometry.axis)[0]
    half = np.concatenate([np.tile(np.arange(len(m.index)) % 2 == 0,m.draws["source"].shape[0])
                           for m in model.members])
    wset = (model.weights, model.weights * half, model.weights * ~half)
    total = np.zeros((4,40))
    areas = np.zeros((len(levels),4))
    for start in range(0,len(uv),128):
        xy, area, radius, angle = polar_nodes(uv[start:start+128],geometry.axis)
        t, _, _ = geometry.locate(xy)
        part = np.where(radius < radii.min(),2,np.where(radius <= radii.max(),3,4))
        hit = t >= 0
        part[hit] = np.where(geometry.inside[t[hit]],0,1)
        groups = 8 * part + np.floor((angle+np.pi/8)/(np.pi/4)).astype(int) % 8
        fields,_ = latent_arrays(model.members,xy)
        v = np.ascontiguousarray(np.concatenate(fields).T)
        order = np.argsort(v,axis=1,kind="stable")
        medians = [np.exp(inverse_cdf(v,w,order=order)[0]) for w in wset]
        for out, x in zip(total, [*medians,np.exp(v) @ model.weights], strict=True):
            out += np.bincount(groups,weights=area*x*1e-5,minlength=40)
        for i,level in enumerate(levels):
            central,even,odd = [m >= level for m in medians]
            union = central | even | odd
            areas[i] += (area @ central, area @ (even != odd), area @ union, (area*area) @ union)
    return total,areas


def median_integral(model, expected_mean, levels):
    streams = [qmc.Sobol(2,scramble=True,seed=seed) for seed in SEEDS]
    acc = np.zeros((len(streams),4,40))
    area_acc = np.zeros((len(streams),len(levels),4))
    previous = None
    history = []
    for power in ORDERS:
        for i,stream in enumerate(streams):
            uv = stream.random_base2(power if previous is None else power-1)
            sums,areas = integrate_points(model,uv,levels)
            acc[i] += sums
            area_acc[i] += areas
        reps = acc/(2**power)
        avg = reps.mean(axis=0)
        central = avg[0].sum()
        spread = abs(reps[0,0]-reps[1,0]).sum()/central
        change = None if previous is None else abs(avg[0]-previous).sum()/central
        split = float(np.max(abs(avg[1:3].sum(axis=1)/central-1)))
        mean_error = abs(avg[3].sum()/expected_mean-1)
        history.append({"points_per_scramble":2**power,"central_km3":float(central),"split":split,
                        "nested":change,"scramble":spread,"independent_mean":mean_error})
        spatial = change is not None and change < .015 and spread < .025 and mean_error < .01
        if spatial:
            break
        previous = avg[0].copy()
    level_checks = []
    for level,a in zip(levels,area_acc.sum(axis=0),strict=True):
        ess = float(a[2]**2/a[3]) if a[3] else 0.
        disagreement = float(a[1]/a[2]) if a[2] else None
        level_checks.append({"level_cm":float(level),"union_ess":ess,"symmetric_difference":disagreement,
                             "median_enclosed_area_km2":float(a[0]/(2*2**power))})
    stable_levels = all(t["symmetric_difference"] < .10 for t in level_checks if t["union_ess"] >= 64)
    return {"volume_km3":float(central),"parts_sectors_km3":avg[0].reshape(5,8).tolist(),
            "history":history,"level_checks":level_checks,"seeds":list(SEEDS),"spatial_passed":bool(spatial),
            "draw_split_passed":split < .025,"level_splits_passed":stable_levels,
            "passed":bool(spatial and split < .025 and stable_levels)}


def estimate(model, data, folder):
    records, mass = {}, []
    for m in model.members:
        values, records[name(m)] = mass_draws(m,folder)
        mass.append(values)
    sizes = [len(a.reshape(-1,5,8)) for a in mass]
    weights = draw_weights(model.model_weights,sizes)
    all_mass = np.concatenate([a.reshape(-1,5,8) for a in mass])
    totals = all_mass.sum(axis=(1,2))
    west = all_mass[:,:,[5,6,7]].sum(axis=(1,2))
    complete = {"volume_km3":interval(totals,weights),"sw_w_nw_km3":interval(west,weights),
                "sw_w_nw_fraction":interval(west/totals,weights),"members":records,
                "all_passed":all(r["passed"] for r in records.values())}
    lo,hi = np.quantile(data.log_value[~data.censored],[.05,.95])
    levels = np.exp(np.linspace(lo,hi,7))
    rounds = []
    for count in (64,128,256):
        thin = packet(model,count)
        expected = np.dot(model.model_weights,[a[:,m.index].sum(axis=(-1,-2)).mean()
                                               for a,m in zip(mass,thin.members,strict=True)])
        result = median_integral(thin,expected,levels)
        rounds.append({"draws_per_chain":count,**result})
        if result["draw_split_passed"] and result["level_splits_passed"]:
            break
    return thin,complete,{"refinements":rounds,**result},mass
