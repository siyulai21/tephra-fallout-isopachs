"""Fit the station-independent footprints to the registered eruptions and matched deletions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .publication import published_datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.posterior import write_fit
from src.empirical_isopach.models.nested.footprints.release import fit_part
from src.empirical_isopach.models.nested.study import write_json


def cases():
    current = datasets()
    previous = {rec["slug"]:rec for rec in published_datasets()}
    out = {rec["slug"]:rec for rec in current}
    for rec in current:
        old = previous[rec["slug"]]
        if old["path"] != rec["path"]:
            out[rec["slug"]+"-published"] = old
    return out


def job(args):
    slug, rec, index, output, design = args
    data, meta = audited_data(rec["path"])
    key, train, test = "full", data, None
    if index is not None:
        key = ("fold-" if design == "geographic" else "quadrant-")+str(index)
        _, train, test = next(p for p in splits(data,meta["vent_lonlat"][1],design) if p[0]==index)
    folder = output/slug
    if design == "quadrants":
        folder.mkdir(exist_ok=True,parents=True)
        if train is None or len(train.xy)<3 or not (~train.censored).any():
            write_json(folder/(key+"-outcome.json"),dict(status="insufficient_training_or_empty_sector",
                       input_sha256=meta["input_sha256"],train=[] if train is None else train.row_ids.tolist(),test=test.row_ids.tolist()))
            return
        r = fit_part(folder,key,train,test,meta)
        write_json(folder/(key+"-outcome.json"),dict(status="estimated",**r))
    else:
        r = write_fit(folder,key,train,test,meta)
    print(slug,key,r["numerics"]["passed"],r["draws"],flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--case",action="append")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--design",choices=("geographic","quadrants"),default="geographic")
    a=p.parse_args()
    selected=cases()
    if a.case:
        selected={k:selected[k] for k in a.case}
    folds=[None,0,1,2,3] if a.design=="geographic" else [0,1,2,3]
    jobs=[(k,r,i,a.output,a.design) for i in folds for k,r in selected.items()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(job,jobs))
