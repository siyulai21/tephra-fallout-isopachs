"""Matched known-volume, contour, and held-compass-sector trials for geographic reach."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import gammainc

from benchmarks.directional import trial
from benchmarks.mass_mixture import SCENARIOS, case
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.continuation import join
from src.empirical_isopach.models.nested.geometry import DIAMETER, positions
from src.empirical_isopach.models.nested.inference import fit
from src.empirical_isopach.models.nested.reach import PROBS, condition
from src.empirical_isopach.models.nested.reach_integrals import integrate, nodes
from src.empirical_isopach.models.nested.study import write_json
from src.probabilistic_isopach.global_volume import EarthDomain

EXTRA = Path("comparisons/nested-isopachs/pinatubo-preview/data/Pinatubo_1991_layer_C.csv")
ONE_SIDE = ("narrow_eastern_fan", "broad_umbrella")


def records(*, published=False):
    from .publication import published_datasets
    catalog = published_datasets() if published else datasets()
    return [*catalog, {"slug":"pinatubo-1991","name":"Pinatubo 1991 Layer C","path":EXTRA}]


def half_survey(name, seed):
    # Identical positive-only site design for a genuine broad umbrella and a
    # narrow fan. No sample in the entire western half of the deposit.
    rng = np.random.default_rng(seed)
    r, theta = np.meshgrid(np.geomspace(30., 650., 8), np.deg2rad(np.linspace(4., 176., 7)))
    r = r.ravel() * np.exp(rng.normal(0., .03, r.size))
    theta = theta.ravel() + rng.uniform(-.03, .03, theta.size)
    axis = EarthDomain.wgs84(0.).axis_scale
    xy = np.column_stack((r * np.sin(theta) * axis, r * np.cos(theta) / axis))
    def length(angle):
        return np.full_like(angle, 130.) if name == "broad_umbrella" else 40. + 190.*np.exp(6.*(np.cos(angle-np.pi/2)-1.))
    def field(points):
        radius, bearing = positions(points, 0.)
        return np.log(100.) - radius / length(bearing)
    angle = (np.arange(65536) + .5) * (2*np.pi/65536)
    scale = length(angle)
    volume = float(2*np.pi*1e-5*100.*np.mean(scale**2*gammainc(2., DIAMETER/scale)))
    y = field(xy) + rng.normal(0., .15, len(xy))
    data = Observations(xy, y, np.zeros(len(y), bool), np.zeros(len(y)), np.arange(len(y))).canonical()
    return data, {"specified":(field, volume)}


def known_design(name, seed):
    if name in ONE_SIDE:
        return half_survey(name, seed)
    data, truth = trial(name, seed)
    if name != "invisible_sector":
        _, total, field = case(name, seed, return_field=True)
        return data.canonical(), {"specified":(field, total)}
    axis = EarthDomain.wgs84(0.).axis_scale
    def field(xy):
        return 5. - positions(xy, 0.)[0]/60.
    def unseen(xy):
        d2 = np.sum((xy * [1/axis, axis] - [100.,100.])**2, axis=1)
        extra = 2000.*np.maximum(1.-d2/50.**2, 0.)**3
        return np.logaddexp(field(xy), np.log(np.maximum(1e-300, extra)))
    return data.canonical(), {"specified":(field,truth["total_km3"]), "hidden":(unseen,truth["same_data_hidden_total_km3"])}


def compact_numerics(num):
    return {"passed":num["passed"], "attempts":[{k:v for k,v in a.items()
            if k not in ("tempering","log_evidence_exponential_power")} for a in num["attempts"]]}


def matched_models(data, lat, seed):
    base, diagnostics = fit(data, lat, initial=512, ceiling=2048, seed=213091)
    parent, num = join(base, data), compact_numerics(diagnostics)
    model, imp = condition(parent, data, seed=seed)
    attempts = [imp]
    source_num = num
    if not imp["passed"]:
        pos = data.subset(~data.censored | data.lower_censored)
        raw, diag = fit(pos, lat, initial=768, ceiling=6144, seed=seed)
        proposal = join(raw, pos)
        model, imp = condition(proposal, data, positive_proposal=True, seed=seed)
        attempts.append(imp)
        source_num = compact_numerics(diag)
    return parent, model, {"reference":num,"source":source_num,"importance":attempts}


def known(job):
    name, seed, out = job
    data, truths = known_design(name, seed)
    previous, model, checks = matched_models(data, 0., 55031 + seed)
    part = model.reduced(1024)
    trials = [integrate(part, data, 14, k) for k in (42983, 84779)]
    volume = np.mean([v[0].sum(axis=1) for v in trials], axis=0)
    q = [np.quantile(previous.volume(), PROBS).tolist(), np.quantile(volume, PROBS).tolist()]
    levels = np.array([1.,10., 100.])
    result = {"scenario":name, "seed":seed, "numerics":checks, "volume_km3":q,
              "method_order":["whole_area_rank", "corridor"], "level_cm":levels.tolist(), "truths":{},
              "volume_by_scramble_km3":[np.quantile(v[0].sum(axis=1),PROBS).tolist() for v in trials],
              "volume_populations_km3":[np.quantile(v,PROBS).tolist() for v in np.array_split(volume,4)],
              "posterior_draws_integrated":len(volume),"nodes_per_scramble":2**14}
    for label,(truth,total) in truths.items():
        runs = []
        for k in (42983, 84779):
            xy,w,_,theta = nodes(14, k, 0.)
            stats = np.zeros((2,len(levels),4))
            for ids in np.array_split(np.arange(len(xy)), 96):
                actual = truth(xy[ids]) >= np.log(levels)[:, None]
                west = np.sin(theta[ids]) < 0.
                for i, m in enumerate((previous.reduced(512), model.reduced(512))):
                    pred = np.median(m.field(xy[ids]), axis=0) >= np.log(levels)[:, None]
                    for j,mask in enumerate((pred & ~actual, actual & ~pred, pred & ~actual & west, actual & ~pred & west)):
                        stats[i,:,j] += mask @ w[ids]
            runs.append(stats / len(xy))
        result["truths"][label] = {"volume_km3":float(total), "coverage":[bool(a<=total<=b) for a,_,b in q],
            "area_statistic_order":["false","missed","false_west","missed_west"],
            "area_km2":np.mean(runs,axis=0).tolist(),"scrambles":[42983,84779]}
    write_json(out/f"{name}-{seed}.json", result)
    print(name, seed, [round(x[1],2) for x in q], flush=True)


def quadrant(job):
    rec, index, out = job
    data, meta = audited_data(rec["path"], allow_intervals=True)
    lat = meta["vent_lonlat"][1]
    _, train, test = list(splits(data, lat, "quadrants"))[index]
    receipt = {"slug":rec["slug"],"quadrant":index,"input_sha256":meta["input_sha256"],
               "test_rows":test.row_ids.tolist(),"training_rows":[] if train is None else train.row_ids.tolist()}
    if train is None or len(train.xy) < 3 or not (~train.censored).any():
        receipt.update(status="empty_or_insufficient", reason="No test observations or insufficient training evidence")
    else:
        previous, model, checks = matched_models(train, lat, 46019+index)
        receipt.update(status="estimated", numerics=checks,
                       previous_log_score=previous.score(test).tolist(), corridor_log_score=model.score(test).tolist())
    write_json(out/f"{rec['slug']}-{index}.json",receipt)
    print(rec["slug"],index,receipt["status"],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--workers",default=1,type=int)
    p.add_argument("--design",choices=("known","quadrants"),required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.design == "known":
        jobs = [(s,k,a.output) for k in (90,91) for s in (*SCENARIOS,"invisible_sector")]
        jobs.extend((s,k,a.output) for k in (1201,1202) for s in ONE_SIDE)
        run = known
    else:
        jobs = [(rec,i,a.output) for i in range(4) for rec in records()]
        run = quadrant
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,jobs))


if __name__ == "__main__":
    main()
