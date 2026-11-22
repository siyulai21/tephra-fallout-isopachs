"""Independent spherical-posterior checks at audited archive horizons."""
import json
from pathlib import Path

import arviz as az
import numpy as np
from scipy.special import log_ndtr,logsumexp
from ...region import SurveyRegion

from ...core.distributions import LogMixture
from ..directional.archive_readings import positions
from ..directional.ensemble.fields import inverse_cdf
from ..directional.study import digest,write_json
from .geometry import log_field,to_sphere
from .inference import flatten


def archive_checks(samples, data, vent, inventory, cutoffs=(.1,1.,10.), *, bearing=None):
    sites = inventory["sites"]
    xy = positions(sites,vent)
    used = {sid for scenario in inventory["scenarios"] for sid in scenario["effective_limit_cm"]}
    for site, point in zip(sites, xy, strict=True):
        if site["id"] in used and np.any(np.linalg.norm(data.xy-point,axis=1)<1e-6):
            raise ValueError("An archive likelihood overlaps an accepted site. Review dependence before combining readings.")
    p = flatten(samples)
    unit = to_sphere(xy,vent[1])
    n = len(p["noise"])
    values = [log_field({k:v[i:i+128] for k,v in p.items()},unit) for i in range(0,n,128)]
    mu = np.concatenate(values)
    noise,eps = p["noise"][:,None],p["contamination"][:,None]
    shape = samples["log_volume"].shape
    volume = np.exp(p["log_volume"])
    multiplicity = 1
    if bearing is not None:
        from .bearings.field import BearingSupport, POWERS, multiplier
        idx,phi = bearing["flat_index"],bearing["azimuth_rad"]
        h,theta = BearingSupport(data,vent[1]).evaluate(xy)
        means = [mu[idx]+np.log(multiplier(h,theta,phi[:,None],power)) for power in POWERS]
        multiplicity = len(means)
        mu = np.stack(means,axis=1).reshape(-1,len(sites))
        noise,eps = (np.repeat(a[idx],multiplicity,axis=0) for a in (noise,eps))
        volume = bearing["volume_km3"].ravel()
        n = len(volume)
        shape = (shape[0],n//shape[0])
    all_models,archive_weights = {},{}
    maps = {s["id"]:i for i,s in enumerate(sites)}
    within = SurveyRegion(data.xy).contains(xy)
    for scenario in [None,*inventory["scenarios"]]:
        key = scenario["id"] if scenario else "quantitative"
        score = np.zeros(n)
        if scenario:
            for sid,limit in scenario["effective_limit_cm"].items():
                z = (np.log(limit)-mu[:,maps[sid]])/noise[:,0]
                with np.errstate(divide="ignore"):
                    score += np.logaddexp(np.log1p(-eps[:,0])+log_ndtr(z),np.log(eps[:,0])+log_ndtr(z/4))
        normalized = score-logsumexp(score)
        weights = np.exp(normalized)
        if bearing is not None and scenario is None:
            q = np.quantile(volume,(.025,.5,.975)).tolist()
        else:
            q = inverse_cdf(volume,weights,(.025,.5,.975)).tolist()
        cluster_mass = weights.reshape(-1,multiplicity).sum(axis=1)
        ess = float(1/(cluster_mass@cluster_mass))
        cluster_score = logsumexp(score.reshape(-1,multiplicity),axis=1)
        pareto = float(az.psislw(cluster_score)[1]) if np.ptp(cluster_score)>1e-12 else 0.
        chain_scores = score.reshape(shape)
        chain_p = np.exp(logsumexp(chain_scores,axis=1)-np.log(shape[1]))
        chain_volume = [float(inverse_cdf(v,np.exp(s-logsumexp(s)))[0])
                        for v,s in zip(volume.reshape(shape),chain_scores,strict=True)]
        # Archive probabilities are bounded above by one. Check concentration, chain influence
        # and actual volume stability before using a self-normalized update as an estimate.
        chain_mass = weights.reshape(shape).sum(axis=1)
        stable = (ess>=400 and pareto<.7 and min(chain_mass)>.10 and max(chain_mass)<.40
                  and max(abs(np.asarray(chain_volume)/q[1]-1))<.15)
        observation = LogMixture(np.r_[weights*(1-eps[:,0]),weights*eps[:,0]],np.vstack([mu,mu]),
                                 np.tile(np.r_[noise[:,0]**2,16*noise[:,0]**2][:,None],(1,len(sites))))
        pred=[{"id":s["id"],"name":s["name"],"in_regional_hull":bool(inside),"latent_cm":{},"new_site_cm":{},
               "p_local_below_scale":{}} for s,inside in zip(sites,within,strict=True)]
        for prob,label in ((.025,"lower95"),(.05,"lower90"),(.5,"median"),(.975,"upper95")):
            latent=np.exp(inverse_cdf(mu.T,weights,[prob])[0]);loc=np.exp(observation.quantile(prob))
            for i,row in enumerate(pred):
                row["latent_cm"][label] = float(latent[i]);row["new_site_cm"][label] = float(loc[i])
        for scale in cutoffs:
            for row,v in zip(pred,observation.cdf(np.log(scale)),strict=True):
                row["p_local_below_scale"][str(scale)] = float(v)
        all_models[key]={"sites":pred,"scenario":scenario,"full_field_volume_km3":dict(zip(("lower95","median","upper95"),q,strict=True)),
                         "joint_reading_probability":float(np.exp(logsumexp(score)-np.log(n))),
                         "chain_reading_probabilities":chain_p.tolist(),"raw_weight_ess":ess,"pareto_k":pareto,
                         "chain_mass":chain_mass.tolist(),"chain_volume_medians_km3":chain_volume,
                         "importance_checks_passed":bool(stable)}
        archive_weights[key]=weights.reshape(shape)
    return all_models,archive_weights


def publish(posterior, data, meta, inventory, output, *, bearing_packet=None):
    if not inventory["sites"]:
        return None
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cutoffs = sorted({.1, 1., 10., *(s["limit_cm"] for s in meta.get("interpreted_upper_limits", []))})
    bearing = None
    if bearing_packet is not None:
        with np.load(bearing_packet) as saved:
            bearing = dict(saved)
    with np.load(posterior) as source:
        info,weights=archive_checks(dict(source),data,meta["vent_lonlat"],inventory,cutoffs,bearing=bearing)
    path=output/"spherical-archive-weights.npz"
    np.savez_compressed(path,**weights)
    report={"posterior_sha256":digest(posterior),"weights_sha256":digest(path),"input_sha256":meta["input_sha256"],
            "method":("Self-normalized archive-likelihood update of complete mass-normalized posterior draws"
                      if inventory["scenarios"] else "Predictions using the original equal weights of the fitted spherical posterior"),
            "diagnostic_policy":"ESS at least 400, Pareto k under 0.7, each chain supplies 10–40% of mass, each chain volume median within 15% of pooled. Raw weights used; instability is retained.",
            "scenarios":info}
    if bearing_packet is not None:
        report["bearing_packet_sha256"] = digest(bearing_packet)
        report["accepted_archive_bounds_cm"] = {s["id"]:float(s["limit_cm"]) for s in meta.get("interpreted_upper_limits",[])}
        report["method"] = "Whole-deposit bearing continuation with any additional inspected-archive likelihood. Effective size groups the three widths of each parent draw."
    for item in report["scenarios"].values():
        item["full_field_volume_km3"].pop("median", None)
    write_json(output/"spherical-archive-checks.json",report)
    return report


def write_summary(report, output):
    if report is None:
        return
    lines = ["# Spherical fallout estimates at inspected archives", "",
             "These whole-Earth volume intervals are conditional on the spherical model and the archive response specified for each row.", "",
             "| Archive interpretation | Conditional 95% interval (km³) | Numerical check |",
             "|---|---:|---|"]
    for result in report["scenarios"].values():
        scenario = result["scenario"]
        title = scenario["title"] if scenario else "Quantitative thickness evidence"
        interval = result["full_field_volume_km3"]
        check = "Passed" if result["importance_checks_passed"] else "Independent refit required"
        lines.append(f"| {title} | {interval['lower95']:,.1f}–{interval['upper95']:,.1f} | {check} |")
    if "bearing_packet_sha256" in report:
        lines += ["", "## Thickness at the inspected horizons", "",
                  "These predictions use the accepted thicknesses and bounds. The final column gives the probability that a new local reading would lie below a fitted bound, conditional on its archive interpretation.", "",
                  "| Site | Median latent thickness (cm) | Conditional 95% (cm) | Interpreted upper bound (cm) | Probability below bound |",
                  "|---|---:|---:|---:|---:|"]
        bounds = report["accepted_archive_bounds_cm"]
        for site in report["scenarios"]["quantitative"]["sites"]:
            q = site["latent_cm"]
            limit = bounds.get(site["id"])
            bound = f"{limit:g}" if limit is not None else "—"
            prob = f"{site['p_local_below_scale'][str(float(limit))]:.1%}" if limit is not None else "—"
            lines.append(f"| {site['name']} | {q['median']:.3g} | {q['lower95']:.3g}–{q['upper95']:.3g} | {bound} | {prob} |")
    if any(result["scenario"] for result in report["scenarios"].values()):
        lines.extend(["", "## Assumptions", ""])
    for result in report["scenarios"].values():
        if result["scenario"]:
            scenario = result["scenario"]
            lines.append(f"- **{scenario['title']}**. {scenario['assumption']}")
    lines.extend(["", "[Site predictions, effective scales and convergence checks](spherical-archive-checks.json) · [Weights of the complete posterior draws](spherical-archive-weights.npz)", ""])
    (Path(output)/"README.md").write_text("\n".join(lines))


def case_archives(path, output, data, meta, supplied=None):
    """Attach the catalogue's inspected sites to the principal spherical case, when available."""
    from ..directional.archive_readings import load_inventory
    inventory, source = load_inventory(path,supplied)
    if not inventory["sites"]:
        return
    output = Path(output)
    folder = output/"archive-evidence"
    report = publish(output/"posterior.npz",data,meta,inventory,folder,bearing_packet=output/"bearing/baseline.npz")
    report["archive_inventory_sha256"] = digest(source)
    write_json(folder/"spherical-archive-checks.json",report)
    write_summary(report,folder)
    page = output/"README.md"
    link = "[Thickness and predictive intervals at inspected archives](archive-evidence/README.md)"
    if link not in page.read_text():
        with page.open("a") as stream:
            stream.write("\n"+link+"\n")
    manifest = output/"artifacts.json"
    files = json.loads(manifest.read_text())
    files["README.md"] = digest(page)
    for name in ("README.md","spherical-archive-checks.json","spherical-archive-weights.npz"):
        files["archive-evidence/"+name] = digest(folder/name)
    write_json(manifest,files)
