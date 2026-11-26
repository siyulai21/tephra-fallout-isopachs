"""Replay the standalone principal Pinatubo fit, interval scores and core survey checks."""
import argparse
from tempfile import TemporaryDirectory
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from benchmarks.verify_deposit_maps import verify_global
from benchmarks.pinatubo_offshore import INPUT, REV, ROOT, read, write
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.archive_readings import positions
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.legacy_posterior import FootprintField, engine, predictions
from src.empirical_isopach.models.nested.inference import log_events
from src.empirical_isopach.models.nested.profile import parameters
from src.empirical_isopach.models.nested.study import digest

OUT = ROOT/"eruptions/pinatubo-1991-layer-c/results/nested-isopachs"


def reproduce(home, output):
    data, meta = audited_data(INPUT)
    verify_global(home, "nested-isopachs", data, meta)
    from src.empirical_isopach.core.catalog import dataset_record
    assert ROOT / dataset_record(INPUT)["principal_nested_result"] == OUT
    frame = read(INPUT)
    packet = home/"footprints"
    receipt=json.loads((packet/"report.json").read_text())
    assert receipt["engine_sha256"]==engine() and receipt["input_sha256"]==digest(INPUT)
    for name,expected in receipt["posterior_sha256"].items():
        assert digest(packet/name)==expected
    for key, result in receipt["fits"].items():
        initial = result["attempts"][0]
        saved = home/"attempts"/"initial"/(key+".npz")
        if saved.is_file():
            assert digest(saved) == initial["posterior_sha256"]
        assert result["attempts"][-1]["posterior_sha256"] == receipt["posterior_sha256"][key+"-selected.npz"]
    model=FootprintField.load(packet/"full-selected.npz")
    heldout=pd.read_csv(packet/"heldout.csv.gz").set_index("row_id")
    assert heldout.index.is_unique and set(heldout.index)==set(data.row_ids)
    assert len(frame)==receipt["accepted_rows"]==len(data.xy)
    for i,train,test in splits(data,meta["vent_lonlat"][1],"geographic"):
        key="fold-"+str(i)
        info=receipt["fits"][key]
        assert info["train"]==train.row_ids.tolist() and info["test"]==test.row_ids.tolist()
        trial=FootprintField.load(packet/(key+"-selected.npz"))
        for score in predictions(trial,test):
            np.testing.assert_allclose(heldout.loc[score["row_id"],"log_score"],score["log_score"],rtol=2e-7,atol=1e-8)
    table=[]
    for pos,row in enumerate(frame):
        if row["Kind"]!="interval":
            continue
        row_id=pos+2
        h=heldout.loc[row_id]
        table.append(dict(id=row["EvidenceID"],site=row["Site"],low_cm=row["Lower_cm"],high_cm=row["Upper_cm"],
                          fold=int(h["fold"]),fold_passed=h["fit_passed"],heldout_probability=float(np.exp(h["log_score"]))))
    assert len(table)==18 and all(0<r["heldout_probability"]<=1 for r in table)
    write(output/"interval-validation.csv",table)

    archive=json.loads((REV/"archive-probes.json").read_text())["sites"]
    records=[]
    for i in range(0,len(archive),10):
        batch=archive[i:i+10]
        field=model.field(positions(batch,meta["vent_lonlat"]))
        for j,site in enumerate(batch):
            lo,med,hi=np.exp(np.quantile(field[:,j],[.025,.5,.975]))
            rec=dict(site_id=site["id"],name=site["name"],reading=site["reading"],lat=site["latitude"],lon=site["longitude"],
                     principal_role=site.get("in_principal_as",""),latent_cm_p025=lo,latent_cm_p50=med,latent_cm_p975=hi)
            for d in (.1,.2,.5,2.):
                rec[f"p_latent_at_least_{d:g}cm"]=np.mean(field[:,j]>=np.log(d))
            rec["visible_fraction_to_hide_median_at_1mm"]=0.1/med
            records.append(rec)
    write(output/"marine-predictions.csv",records)

    # Reweight the principal draws by the inverse likelihood of entire linked
    # source groups. These comparisons use coherent volumes and keep the model fixed.
    volume=model.volume()
    noise=parameters(model.z)[-1]
    omissions=[]
    for name,ids in {
        "four_SO95_ceilings":[r["EvidenceID"] for r in frame if r["EvidenceID"].startswith("SO95-")],
        "SO132_47_reconstruction":["W09-47"],
        "SO132_47_replace_with_2cm_ceiling":["W09-47"],
        "SO114_4_reconstruction":["W08-404"],
        "farm_recollections":[r["EvidenceID"] for r in frame if r["EvidenceID"].startswith("Q20-A")],
        "David_pumice":[r["EvidenceID"] for r in frame if r["EvidenceID"].startswith("D96-")],
        "Quimbo_pits":[r["EvidenceID"] for r in frame if r["EvidenceID"].startswith("Q20-P")],
        "trap_equivalent":["WW-SCS-C"],
        "Wetzel_reconstruction":["W09-33"],
    }.items():
        part=data.subset(np.isin(data.row_ids,[i for i,r in enumerate(frame,2) if r["EvidenceID"] in ids]))
        assert len(part.xy) == len(ids) and ids, name
        mu=model.field(part.xy)
        logw=-log_events(mu,noise,part).sum(axis=1)
        if name == "SO132_47_replace_with_2cm_ceiling":
            ceiling = replace(part, log_value=np.log([2.0]), censored=np.array([True]))
            logw += log_events(mu,noise,ceiling).sum(axis=1)
        wt=np.exp(logw-logsumexp(logw))
        ess=1/np.sum(wt**2)
        order=np.argsort(volume)
        quantiles=np.interp([.025,.5,.975],np.cumsum(wt[order]),volume[order])
        ok=ess>=100 and ess/len(wt)>=.05
        omissions.append(dict(group=name,source_ids="|".join(ids),ess=ess,draws=len(wt),weight_check_passed=bool(ok),
                     conditional_draw_median_km3=quantiles[1] if ok else "",p025_km3=quantiles[0] if ok else "",p975_km3=quantiles[2] if ok else ""))
    write(output/"conditional-omissions.csv",omissions)
    assert digest(packet/"support.json.gz")==receipt["support_sha256"] and digest(packet/"heldout.csv.gz")==receipt["heldout_sha256"]
    np.testing.assert_allclose(np.quantile(volume,[.025,.975]),receipt["conditional_interval95_km3"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=OUT)
    p.add_argument("--verify", action="store_true", help="Compare the three published diagnostic tables with a fresh replay")
    args=p.parse_args()
    if args.verify:
        with TemporaryDirectory() as temporary:
            output=Path(temporary)
            reproduce(args.folder, output)
            for item in output.glob("*.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(item), pd.read_csv(args.folder/item.name),
                                              check_dtype=False, rtol=2e-7, atol=1e-8)
    else:
        reproduce(args.folder, args.folder)
