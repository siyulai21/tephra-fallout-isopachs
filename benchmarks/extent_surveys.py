"""Transfer the nested continuation to four previously frozen survey or archive comparisons."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .nested_reach_field import FIELD_CASES, partition
from src.empirical_isopach.models.extent_review import ExtentFields
from src.empirical_isopach.models.extent_validation import scores
from src.empirical_isopach.models.extent_weights import calibration
from src.empirical_isopach.models.nested.reach import ReachDraws
from src.empirical_isopach.visualization.deposit import read_json,digest


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for slug in FIELD_CASES:
        folder=Path("comparisons/nested-reach/field-surveys")/slug
        report=read_json(folder/"validation.json")
        if digest(folder/"corridor.npz")!=report["corridor_packet_sha256"]:
            raise ValueError("Frozen survey posterior differs")
        train,test,meta,_=partition(slug,output)
        identity=report["identity"]
        if train.row_ids.tolist()!=identity["training_rows"] or test.row_ids.tolist()!=identity["test_rows"]:
            raise ValueError("Survey partition differs")
        target="pinatubo-1991-layer-c" if slug=="pinatubo-1991" else slug
        cal=calibration("nested-isopachs",meta["input_sha256"],exclude_slug=target)
        parent=ReachDraws.load(folder/"corridor.npz")
        fields=ExtentFields("nested-isopachs",parent,train,meta["vent_lonlat"][1])
        vals=scores(fields,test)
        w=np.asarray(cal["weights"])
        lp=logsumexp(np.log(w[:,None])+[vals[s+"_log_score"] for s in cal["states"]],axis=0)
        cdf=w@np.stack([vals[s+"_cdf"] for s in cal["states"]])
        for i,rid in enumerate(test.row_ids):
            rows.append(dict(survey=slug,input_sha256=meta["input_sha256"],posterior_sha256=report["corridor_packet_sha256"],
                row_id=rid,exact=not test.censored[i],parent_log_score=vals["parent_log_score"][i],principal_log_score=lp[i],
                parent_cdf=vals["parent_cdf"][i],principal_cdf=cdf[i],fit_passed=report["revised_source_passed"]))
    pd.DataFrame(rows).to_csv(output/"external-surveys.csv",index=False,lineterminator="\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args().output)
