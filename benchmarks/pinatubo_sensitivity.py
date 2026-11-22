"""Test how the Pinatubo nested corridors respond to source selection and marine intervals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .nested_reach_field import assess
from .pinatubo import AUDIT, INPUT, src
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.observations import read_observations
from src.empirical_isopach.models.nested.continuation import check_and_refine
from src.empirical_isopach.models.nested.inference import fit
from src.empirical_isopach.models.nested.reach import condition
from src.empirical_isopach.models.nested.reach_fit import signature
from src.empirical_isopach.models.nested.reach_integrals import checked
from src.empirical_isopach.models.nested.study import digest, write_json


def marine_intervals(folder):
    frame=pd.read_csv(INPUT)
    frame["Lower_cm"],frame["Upper_cm"]=np.nan,np.nan
    paired=[]
    for r in src("hess-table-1.csv"):
        ident="H01-"+r["site"].zfill(2)
        mask=frame.EvidenceID.eq(ident)
        if mask.any() and "–" in r["thickness_mm"]:
            low,high=[float(v)/10 for v in r["thickness_mm"].split("–")]
            assert frame.loc[mask,"Kind"].iloc[0]=="lower_limit" and frame.loc[mask,"Limit_cm"].iloc[0]==low
            frame.loc[mask,["Kind","Limit_cm","Lower_cm","Upper_cm"]]=["interval",np.nan,low,high]
            paired.append(ident)
    path=folder/"interval-input.csv"
    frame.to_csv(path,index=False,lineterminator="\n")
    data,_=read_observations(path,allow_intervals=True)
    return data,paired,digest(path)


def scenario(name, root):
    folder=root/name
    folder.mkdir(parents=True,exist_ok=False)
    full,meta=audited_data(INPUT)
    lookup=pd.read_csv(AUDIT/"selection.csv").dropna(subset=["accepted_csv_row"]).set_index("accepted_csv_row")
    source=lookup.loc[full.row_ids,"source"].to_numpy()
    identities=lookup.loc[full.row_ids,"id"].to_numpy()
    train=full
    test=None
    alternate={}
    if name=="marine-holdout":
        withheld=np.isin(source,["WW1","H01","W04","S99"])
    elif name=="tabulated-thickness":
        withheld=np.isin(source,["P7","K3"])
    elif name=="marine-intervals":
        train,paired,sha=marine_intervals(folder)
        alternate={"interval_rows":paired,"interval_input_sha256":sha}
    else:
        raise ValueError(name)
    if name!="marine-intervals":
        train,test=full.subset(~withheld),full.subset(withheld)
    latitude=meta["vent_lonlat"][1]
    raw,internal=fit(train,latitude,initial=768,ceiling=6144)
    _,previous,checks=check_and_refine(raw,train,internal,ceiling=12288)
    model,importance=condition(previous,train,seed=71183)
    attempts=[importance]
    source_passed=checks["passed"]
    numerical={"initial":internal,"parent":checks}
    if not importance["passed"]:
        pos=train.subset(~train.censored|train.lower_censored)
        base,count=fit(pos,latitude,initial=768,ceiling=12288,seed=71183)
        _,proposal,positive=check_and_refine(base,pos,count,ceiling=12288,seed=71183)
        model,importance=condition(proposal,train,positive_proposal=True,seed=71183)
        attempts.append(importance)
        source_passed=positive["passed"]
        numerical["positive_proposal"]=positive
    integral,_=checked(model,train)
    model.save(folder/"corridor.npz")
    report={"case":name,"input_sha256":meta["input_sha256"],"engine_sha256":signature(),
            "train_rows":train.row_ids.tolist(),"test_rows":[] if test is None else test.row_ids.tolist(),
            "corridor_packet_sha256":digest(folder/"corridor.npz"),"imp_attempts":attempts,**alternate,
            "numerics":numerical,"volume":integral,"passed":bool(source_passed and importance["passed"] and
                integral["quadrature"][-1]["passed"] and integral["volume_populations_passed"])}
    if test is not None:
        score=assess(model,test)
        report["validation"]={k:v for k,v in score.items() if k not in ("row_log_score","latent_log_cm","predictive_cdf_at_value")}
        frame=pd.DataFrame({"row_id":test.row_ids,"id":identities[withheld],"source":source[withheld],
              "observed_value_or_bound_cm":np.exp(test.log_value),"kind":lookup.loc[test.row_ids,"kind"].to_numpy(),
              "median_predicted_cm":np.exp(score["latent_log_cm"]),"log_score":score["row_log_score"],
              "predictive_cdf_at_observed":score["predictive_cdf_at_value"]})
        frame.to_csv(folder/"heldout.csv",index=False,lineterminator="\n")
    write_json(folder/"result.json",report)
    print(name,len(train.xy),integral["median_map_volume_km3"],report["passed"],flush=True)


def regional_endpoints(root):
    """Keep the sites fixed and use the other recorded inequality from each range."""
    from src.empirical_isopach.reporting import compare_dataset

    folder = root / "regional-upper-endpoints"
    folder.mkdir(parents=True, exist_ok=False)
    _, paired, _ = marine_intervals(folder)
    frame = pd.read_csv(folder / "interval-input.csv")
    bracket = frame["Kind"].eq("interval")
    assert frame.loc[bracket, "EvidenceID"].tolist() == paired
    frame.loc[bracket, "Kind"] = "upper_limit"
    frame.loc[bracket, "Limit_cm"] = frame.loc[bracket, "Upper_cm"]
    path = folder / "upper-endpoints.csv"
    frame.drop(columns=["Lower_cm", "Upper_cm"]).to_csv(path, index=False, lineterminator="\n")
    compare_dataset(path, folder / "regional", name="Pinatubo Layer C, marine upper endpoints", progress=print)


if __name__ == "__main__":
    cases = ["marine-holdout", "tabulated-thickness", "marine-intervals", "regional-upper-endpoints"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=cases)
    args = parser.parse_args()
    for selected in ([args.case] if args.case else cases):
        if selected == "regional-upper-endpoints":
            regional_endpoints(args.output)
        else:
            scenario(selected, args.output)
