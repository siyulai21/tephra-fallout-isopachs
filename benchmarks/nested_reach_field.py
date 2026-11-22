"""Frozen-model prediction on genuine field surveys and transcribed archive bounds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import ndtr

from .nested_reach import case_home
from .nested_reach_challenge import records
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.nested.continuation import check_and_refine, load_packet
from src.empirical_isopach.models.nested.inference import fit
from src.empirical_isopach.models.nested.profile import parameters
from src.empirical_isopach.models.nested.reach import PROBS, condition
from src.empirical_isopach.models.nested.reach_fit import signature
from src.empirical_isopach.models.nested.reach_integrals import checked
from src.empirical_isopach.models.nested.study import digest, write_json
from src.empirical_isopach.validation import automatic_buffer

ARCHIVE = Path("data/sources/nested-reach/calbuco2015.zip")
ZIP_SHA = "9b271e1a114167fca6e41bac14cc088b1783913b7a22f13e092041399bc619ac"
FIELD_CASES = ("calbuco-2015", "aso-4", "b-tm", "pinatubo-1991")


def calbuco(out):
    if digest(ARCHIVE) != ZIP_SHA:
        raise ValueError("Calbuco field archive differs from the released Zenodo source")
    rows = []
    ledger = []
    with ZipFile(ARCHIVE) as z:
        for tag, member in (("training","grl54177.csv"),("validation","reckziegel.csv")):
            raw = pd.read_csv(BytesIO(z.read("calbuco2015-master/POST_DATA/" + member)))
            for i, v in enumerate(raw.itertuples(index=False), 2):
                lo, hi = v.thickness_min, v.thickness_max
                reason, kind = "accepted", "measurement" if lo == hi else "interval"
                if lo == hi == 0.:
                    reason = "recorded_zero_without_site_specific_detection_limit"
                if lo < 0 or hi < lo or (lo == 0 < hi):
                    raise ValueError(f"Review bounds in {member} row {i}")
                duplicates = raw.iloc[:i-2]
                same = ((duplicates.latitude == v.latitude) & (duplicates.longitude == v.longitude)
                        & (duplicates.thickness_min == lo) & (duplicates.thickness_max == hi))
                if same.any():
                    reason = "identical_coordinates_and_reading_counted_once"
                record = {"survey":tag, "member":member, "source_row":i, "label":v.label, "Lon":v.longitude,
                          "Lat":v.latitude,"Lower_cm":lo,"Upper_cm":hi,"decision":reason}
                if reason == "accepted":
                    row = dict(record, Thickness=lo if kind=="measurement" else None, Kind=kind,
                               **{"Vent Lon":-72.618 if not rows else None, "Vent Lat":-41.330 if not rows else None})
                    record["accepted_csv_row"] = len(rows)+2
                    rows.append(row)
                ledger.append(record)
    frame = pd.DataFrame(rows)
    path = out/"field-observations.csv"
    frame.to_csv(path,index=False,lineterminator="\n")
    pd.DataFrame(ledger).to_csv(out/"row-decisions.csv",index=False,lineterminator="\n")
    data,meta = audited_data(path,allow_intervals=True)
    groups = frame.survey.to_numpy()[data.row_ids-2]
    return data.subset(groups=="training"),data.subset(groups=="validation"),meta,path


def partition(slug,out):
    if slug=="calbuco-2015":
        return calbuco(out)
    rec = next(r for r in records() if r["slug"]==slug)
    path = rec["path"]
    data,meta = audited_data(path,allow_intervals=True)
    if slug=="b-tm":
        ids = [x["row_id"] for x in meta["interpreted_upper_limits"]]
        test = np.isin(data.row_ids,ids)
        train = ~test
    elif slug=="pinatubo-1991":
        source = pd.read_csv(case_home(slug)/"data/audit/selection.csv").dropna(subset=["accepted_csv_row"])
        test = np.isin(data.row_ids,source.loc[source.source=="WW1","accepted_csv_row"])
        train = ~test
    else:
        review = pd.read_csv("data/audits/Aso-4/record-review.csv").set_index("csv_row").loc[data.row_ids]
        csv = pd.read_csv(path).iloc[data.row_ids-2]
        np.testing.assert_allclose(csv[["Lon","Lat"]],review[["reviewed_longitude","reviewed_latitude"]],atol=1e-6,rtol=0)
        setting = review.setting.fillna("").str.strip()
        train = setting.isin(("Subaerial","Boring Core, Subaerial","Kyushu, Yamaguchi")).to_numpy()
        test = setting.str.startswith("Submarine").to_numpy()
    return data.subset(train), data.subset(test),meta,path


def assess(model, data):
    mu = model.field(data.xy)
    sigma = parameters(model.z)[-1][:,None]
    sd, wide = np.hypot(sigma,data.log_sd),np.hypot(3.*sigma,data.log_sd)
    def cdf(value):
        return (.95*ndtr((value-mu)/sd)+.05*ndtr((value-mu)/wide)).mean(axis=0)
    latent = np.median(mu,axis=0)
    ranks = cdf(data.log_value)
    exact = ~data.censored
    upper = getattr(data,"interval_log_upper",np.full(len(exact),np.nan))
    scores = model.score(data)
    result = {"row_log_score":scores.tolist(),"negative_log_score":float(-scores.mean()),
              "latent_log_cm":latent.tolist(),"predictive_cdf_at_value":ranks.tolist(),
              "median_absolute_log_error":float(np.median(abs(latent[exact]-data.log_value[exact]))) if exact.any() else None,
              "covered95_exact":int(np.sum(exact & (ranks>=.025)&(ranks<=.975))),"thresholds":{}}
    for cm in (.01,.05,.1,.2,.4,1.,10.):
        t = np.log(cm)
        inside = (exact | data.lower_censored) & (data.log_value >= t)
        outside = ((~data.lower_censored) & (data.log_value < t)) | (np.isfinite(upper) & (upper < t))
        known = inside | outside
        p = 1.-cdf(t)
        hit = latent >= t
        result["thresholds"][str(cm)] = {"adjudicable":int(known.sum()),"measured_inside":int(inside.sum()),
                    "false_exceedances":int(np.sum(hit & outside)), "missed_exceedances":int(np.sum(~hit & inside)),
                    "predictive_brier":float(np.mean((p[known]-inside[known])**2)) if known.any() else None}
    return result


def run(job):
    slug, root = job
    out = root/slug
    out.mkdir(exist_ok=True,parents=True)
    train,test,meta,path = partition(slug,out)
    lat = meta["vent_lonlat"][1]
    packet=out/"training-parent.npz"
    ref=out/"training-parent.json"
    ident={"input_sha256":meta["input_sha256"],"engine":signature(),
           "training_rows":train.row_ids.tolist(),"test_rows":test.row_ids.tolist()}
    if ref.exists():
        receipt=json.loads(ref.read_text())
        if ident!=receipt["identity"] or digest(packet)!=receipt["sha256"]:
            raise ValueError("Independent survey fit identity changed. Use a fresh directory")
        previous=load_packet(packet)
        num=receipt["numerics"]
    else:
        raw, internal = fit(train,lat,initial=768,ceiling=6144)
        _,previous,checks = check_and_refine(raw,train,internal,ceiling=12288)
        np.savez_compressed(packet,**vars(previous))
        num = {"passed":checks["passed"],"parent_passed":checks["parent_passed"],"continued_attempts":checks["continued_attempts"]}
        write_json(ref,{"identity":ident,"sha256":digest(packet),"numerics":num})
    revised, imp = condition(previous,train,seed=71183)
    imps=[imp]
    source_passed=num["passed"]
    if not imp["passed"]:
        pos=train.subset(~train.censored|train.lower_censored)
        raw, num_pos=fit(pos,lat,initial=768,ceiling=12288,seed=71183)
        _, proposal, positive_check=check_and_refine(raw,pos,num_pos,seed=71183,ceiling=12288)
        source_passed=positive_check["passed"]
        revised,imp=condition(proposal,train,positive_proposal=True,seed=71183)
        imps.append(imp)
    revised.save(out/"corridor.npz")
    distances=cKDTree(train.xy).query(test.xy)[0]
    both={key:assess(m,test) for key,m in (("whole_area",previous),("corridor",revised))}
    buffer=automatic_buffer(train)[0]
    far=distances>buffer
    distant={"training_only_buffer_km":float(buffer),"test_rows":test.row_ids[far].tolist(),
             "scores":{key:float(-np.mean(np.asarray(x["row_log_score"])[far])) for key,x in both.items()} if far.any() else {}}
    relative=("field-observations.csv" if slug=="calbuco-2015" else str(Path(path).resolve().relative_to(Path.cwd())))
    report={"identity":ident,"source":relative,"geographically_separated":distant,"q95_coverage_denominator":int((~test.censored).sum()),
            "nearest_training_projected_km":distances.tolist(),"observed_or_limit_cm":np.exp(test.log_value).tolist(),
            "observed_is_bound":test.censored.tolist(),"tested_models":both,"numerics":num,"importance":imps,
            "revised_source_passed":source_passed,"corridor_packet_sha256":digest(out/"corridor.npz")}
    if slug=="calbuco-2015":
        integrated, _=checked(revised,train)
        report["whole_area_volume_km3"]=np.quantile(previous.volume(),PROBS).tolist()
        report["corridor_integration"]=integrated
        report["field_archive_sha256"]=ZIP_SHA
        report["field_decisions_sha256"]=digest(out/"row-decisions.csv")
        from .nested_reach_transects import compare
        compare(out)
    write_json(out/"validation.json",report)
    print(slug,len(train.xy),len(test.xy),[v["negative_log_score"] for v in both.values()],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,[(slug,a.output) for slug in FIELD_CASES]))


if __name__=="__main__":
    main()
