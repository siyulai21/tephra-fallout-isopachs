"""An exactly observation-invariant proximal ash sensitivity on the physical Earth."""
from pathlib import Path
import json

import numpy as np
from shapely.geometry import Point

from .estimate import interval
from .geometry import Geometry, polar
from .inference import field, flat
from .study import digest, write_json


def cap_kernel(radius, nearest):
    if nearest == 0:
        return np.zeros_like(radius,dtype=float)
    return np.maximum(0.,1.-(radius/nearest)**2)**3


def cap_volumes(data, geometry, draws):
    """Positive contrast between the background source and the local vent field."""
    radii, _ = polar(data.xy, geometry.axis)
    nearest = float(np.min(radii))
    if np.any(cap_kernel(radii,nearest)):
        raise ArithmeticError("The proximal sensitivity must vanish at every observation")
    p = flat(draws)
    source = np.exp(field(p,geometry,[[0.,0.]])[:,0])
    addition = np.maximum(0.,np.exp(p["source"])-source) if nearest > 0 else np.zeros_like(source)
    # In source-centred equal-area coordinates, integrate 2*pi*r*(1-(r/a)^2)^3.
    extra = np.pi*nearest**2/4*addition*1e-5
    return nearest, source, addition, extra.reshape(draws["source"].shape)


def peak_candidates(data, g, draws, nearest, amplitude, grid_maximum):
    p = flat(draws)
    index = np.linspace(0,len(p["source"])-1,min(512,len(p["source"]))).astype(int)
    samples = {k:v[index] for k,v in p.items()}
    angles = np.linspace(0.,2*np.pi,128,endpoint=False)
    radii = np.linspace(0.,nearest,49)
    x = np.sin(angles)[None,:]*radii[:,None]*g.axis
    y = np.cos(angles)[None,:]*radii[:,None]/g.axis
    points = np.unique(np.vstack((data.xy,g.triangles.reshape(-1,2),[[0.,0.]],
                                  [grid_maximum],np.c_[x.ravel(),y.ravel()])),axis=0)
    radius, _ = polar(points,g.axis)
    shape = cap_kernel(radius,nearest)
    reference, with_cap = [], []
    for start in range(0,len(points),256):
        selection = slice(start,start+256)
        z = field(samples,g,points[selection])
        reference.extend(z.mean(0))
        extra = amplitude[index,None]*shape[selection]
        with np.errstate(divide="ignore"):
            with_cap.extend(np.logaddexp(z,np.log(extra)).mean(0))

    def describe(values):
        i = int(np.argmax(values))
        xy = points[i]
        triangle = g.locate([xy])[0][0]
        region = ("retained measurement triangles" if g.mesh.geometry.covers(Point(xy))
                  else "transition" if triangle >= 0 else "background continuation")
        return {"xy_km":xy.tolist(),"source_chord_km":float(radius[i]),"geometric_mean_cm":float(np.exp(values[i])),
                "region":region,"nearest_observation_km":float(np.linalg.norm(data.xy-xy,axis=1).min())}
    return {"reference":describe(reference),"with_cap":describe(with_cap),"candidate_count":len(points),
            "rule":"Original mapped maximum, vent, all accepted sites and triangle vertices, plus 49 radii and 128 bearings inside the nearest accepted radius. Finite search on 512 posterior fields."}



def heldout(data, g, work, latitude, validation):
    from scipy.special import log_ndtr, logsumexp

    output = {}
    for kind, record in validation.items():
        scores = np.asarray(record["predictions"]["log_score"],dtype=float)
        changes = []
        for fold in record["fits"]:
            if fold["status"] != "estimated":
                continue
            train = data.subset(np.isin(data.row_ids,fold["training_row_ids"]))
            test = np.flatnonzero(np.isin(data.row_ids,fold["withheld_row_ids"]))
            radius, _ = polar(data.xy[test],g.axis)
            nearest = polar(train.xy,g.axis)[0].min()
            selected = test[radius < nearest]
            if not len(selected):
                continue
            geometry = Geometry(train,latitude)
            attempt = fold["attempts"][-1]
            path = work/kind/f"fold-{fold['fold']}"/attempt["file"]
            if digest(path) != attempt["posterior_sha256"]:
                raise ValueError("Changed geographic posterior for source-cap validation")
            with np.load(path) as f:
                draws = dict(f)
            _, _, amplitude, _ = cap_volumes(train,geometry,draws)
            params = flat(draws)
            test_data = data.subset(selected)
            r, _ = polar(test_data.xy,g.axis)
            change = amplitude[:,None]*cap_kernel(r,nearest)
            with np.errstate(divide="ignore"):
                mu = np.logaddexp(field(params,geometry,test_data.xy),np.log(change))
            _, _, weights = geometry.locate(test_data.xy)
            site_var = 1+geometry.policy.local_sd_ratio**2*np.maximum(0.,1-(weights**2).sum(1))
            sd = np.hypot(np.exp(params["log_noise"])[:,None]*np.sqrt(site_var),test_data.log_sd)
            z = (test_data.log_value-mu)/sd
            lp = np.where(test_data.censored,log_ndtr(np.where(test_data.lower_censored,-z,z)),
                          -.5*z*z-.5*np.log(2*np.pi)-np.log(sd))
            new_scores = logsumexp(lp,axis=0)-np.log(len(mu))
            for index, score in zip(selected,new_scores):
                changes.append({"fold":fold["fold"],"row_id":int(data.row_ids[index]),
                                "original_log_score":float(scores[index]),"with_cap_log_score":float(score)})
                scores[index] = score
        labels = np.asarray(record["folds"])
        by_fold = [-scores[(labels==fold)&np.isfinite(scores)].mean() for fold in range(4)
                   if np.any((labels==fold)&np.isfinite(scores))]
        output[kind] = {"equal_region_log_score":float(np.mean(by_fold)) if by_fold else None,
                        "reference_score":record["equal_region_log_score"],"inside_training_cap":changes,
                        "all_estimable_passed":record["all_estimable_passed"]}
    return output


def publish(data, g, draws, folder, report, work, latitude):
    import arviz as az

    folder = Path(folder)
    base = report["fits"]["reference"]
    nearest, source, amplitude, extra = cap_volumes(data,g,draws)
    with np.load(folder/"volume-draws.npz") as f:
        totals = f["reference"]+extra
    posterior = az.from_dict(posterior={"log_volume":np.log(totals)})
    rhat = float(az.rhat(posterior).log_volume)
    bulk = float(az.ess(posterior,method="bulk").log_volume)
    tail = float(az.ess(posterior,method="tail").log_volume)
    item = {"source_sha256":digest(__file__), "nearest_accepted_chord_km":nearest,
            "nearest_row_ids":data.row_ids[np.isclose(polar(data.xy,g.axis)[0],nearest,rtol=0,atol=1e-9)].tolist(),
            "observed_thickness_change_cm":0., "reference_vent_cm":interval(source),
            "extra_source_cm":interval(amplitude),"extra_km3":interval(extra), "volume_km3":interval(totals),
            "rhat":rhat,"bulk_ess":bulk,"tail_ess":tail,"passed":bool(base["passed"] and rhat<1.01 and min(bulk,tail)>=400),
            "peaks":peak_candidates(data,g,draws,nearest,amplitude,report["map"]["grid_maximum_xy_km"]),
            "assumption":"Add the positive part of background-minus-local source thickness with a cubed parabolic taper to the nearest accepted site. One universal unobserved proximal scenario."}
    item["validation"] = heldout(data,g,Path(work),latitude,report["validation"])
    report["unobserved_proximal_cap"] = item
    write_json(folder/"report.json",report)
    np.savez_compressed(folder/"source-cap.npz",extra_km3=extra,volume_km3=totals)
    peaks = item["peaks"]
    direct = peaks["reference"]
    v = item["volume_km3"]
    geo = item["validation"]["geographic"]
    score = geo["equal_region_log_score"]
    validation_text = (f" Its matched geographic score is {score:.3f}, with {len(geo['inside_training_cap'])} held-out rows inside a training cap."
                       if score is not None else "")
    block = ("## Unmeasured proximal fallout\n\n"
        f"The nearest accepted site is {nearest:,.1f} km from the vent. A compact positive ash cap filling a source depression adds a median "
        f"**{item['extra_km3']['median']:.2f} km³** and gives **{v['median']:,.1f} [{v['lower95']:,.1f}–{v['upper95']:,.1f}] km³**. "
        "It vanishes exactly at every accepted measurement and bound. Its amplitude comes from the fitted source-background contrast and its mass is analytic. "
        f"The cap-inclusive numerical checks {'passed' if item['passed'] else 'carry a warning'}. Other unobserved proximal deposits can differ.{validation_text}\n\n"
        f"Adding the measured sites, triangle vertices and a finer near-source search to the displayed grid gives a reference thickness maximum of "
        f"{direct['geometric_mean_cm']:.1f} cm, {direct['source_chord_km']:.1f} km from the vent in {direct['region']}. "
        f"With the cap the highest candidate is {peaks['with_cap']['source_chord_km']:.1f} km from the vent. Both peak searches are finite. "
        "[Paired volume draws](source-cap.npz).\n\n")
    page = folder/"README.md"
    text = page.read_text()
    start, end = "## Unmeasured proximal fallout","## Assumptions and validation"
    if start in text:
        text = text[:text.index(start)]+text[text.index(end):]
    page.write_text(text.replace(end,block+end))
    manifest = json.loads((folder/"artifacts.json").read_text())
    manifest.update({f:digest(folder/f) for f in ("README.md","report.json","source-cap.npz")})
    write_json(folder/"artifacts.json",manifest)
    return report
