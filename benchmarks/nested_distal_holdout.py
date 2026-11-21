"""Hold out the farthest fifth of sites under the same rule for both nested profiles."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.geometry import positions
from src.empirical_isopach.models.nested.inference import fit
from src.empirical_isopach.models.nested.continuation import check_and_refine
from src.empirical_isopach.models.nested.study import write_json
from src.empirical_isopach.validation import automatic_buffer


def trial(job):
    path, name, slug, output = job
    data,meta = audited_data(path,allow_intervals=True)
    latitude = meta["vent_lonlat"][1]
    distance = positions(data.xy,latitude)[0]
    cut = np.quantile(distance,.8)
    test = data.subset(distance>=cut)
    buffer = automatic_buffer(data)[0]
    gaps = cKDTree(test.xy).query(data.xy)[0]
    train = data.subset((distance<cut)&(gaps>=buffer))
    result = {"name":name,"input_sha256":meta["input_sha256"],"cutoff_authalic_km":float(cut),
              "buffer_km":buffer,"training_rows":train.row_ids.tolist(),"test_rows":test.row_ids.tolist(),"fits":{}}
    for key,budget in (("initial",False),("distal_budget",True)):
        model,numerics = fit(train,latitude,budget=budget)
        scores = model.score(test)
        result["fits"][key] = {"equal_site_negative_log_score":float(-scores.mean()),
                               "site_log_scores":scores.tolist(),"numerics":numerics,
                               "conditional_volume_km3":np.quantile(model.volume(),[.025,.5,.975]).tolist()}
        if budget:
            model,revised,quality = check_and_refine(model,train,numerics)
            parent_scores = model.score(test)
            result["fits"][key].update(equal_site_negative_log_score=float(-parent_scores.mean()),site_log_scores=parent_scores.tolist(),
                        conditional_volume_km3=np.quantile(model.volume(),[.025,.5,.975]).tolist(),numerics=quality)
            scores = revised.score(test)
            result["fits"]["continued"] = {"equal_site_negative_log_score":float(-scores.mean()),
                       "site_log_scores":scores.tolist(),"population_check":quality["continuation"],
                       "parent_passed":quality["parent_passed"],
                       "conditional_volume_km3":np.quantile(revised.volume(),[.025,.5,.975]).tolist()}
    write_json(output/(slug+".json"),result)
    print(name,[v["equal_site_negative_log_score"] for v in result["fits"].values()],flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--extra",type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    jobs = [(r["path"],r["name"],r["slug"],a.output) for r in datasets()]
    if a.extra:
        jobs.append((a.extra,a.extra.stem,"unregistered",a.output))
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(trial,jobs))
