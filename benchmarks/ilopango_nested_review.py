"""Replay the Unit G principal release, its interval scores and field sensitivities."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .ilopango_field_review import INPUT, OUT, REV, VENT, GEOD, read
from .verify_deposit_maps import verify_global
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.archive_readings import positions
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.footprints.legacy_posterior import FootprintField, engine, predictions
from src.empirical_isopach.models.nested.inference import log_events
from src.empirical_isopach.models.nested.profile import parameters
from src.empirical_isopach.models.nested.study import digest


def quantiles(values, weights, levels):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) - weights[order] / 2
    return np.interp(levels, cumulative, values[order])


def reach(field, weights, distance):
    medians = np.array([quantiles(column, weights, 0.5) for column in field.T])
    above = np.flatnonzero(medians >= np.log(0.1))
    if not len(above) or above[-1]+1 == len(distance):
        return None
    k = above[-1]
    return np.interp(np.log(0.1), medians[k:k+2][::-1], distance[k:k+2][::-1])


def reproduce(home, output):
    data, meta = audited_data(INPUT)
    verify_global(home, "nested-isopachs", data, meta)
    packet = home/"footprints"
    receipt = json.loads((packet/"report.json").read_text())
    assert receipt["engine_sha256"] == engine() and receipt["input_sha256"] == digest(INPUT)
    for filename, checksum in receipt["posterior_sha256"].items():
        assert digest(packet/filename) == checksum
    for key, fit in receipt["fits"].items():
        if len(fit["attempts"]) > 1:
            assert digest(home/"attempts/initial"/(key+".npz")) == fit["attempts"][0]["posterior_sha256"]
    assert digest(packet/"support.json.gz") == receipt["support_sha256"]
    assert digest(packet/"heldout.csv.gz") == receipt["heldout_sha256"]
    frame = read(INPUT)
    by_id = {r["EvidenceID"]: (i, r) for i,r in enumerate(frame,2)}
    held = pd.read_csv(packet/"heldout.csv.gz").set_index("row_id")
    assert held.index.is_unique and set(held.index) == set(data.row_ids)
    for i, train, test in splits(data, meta["vent_lonlat"][1], "geographic"):
        key = f"fold-{i}"
        assert receipt["fits"][key]["test"] == test.row_ids.tolist()
        assert receipt["fits"][key]["train"] == train.row_ids.tolist()
        trial = FootprintField.load(packet/(key+"-selected.npz"))
        for entry in predictions(trial,test):
            np.testing.assert_allclose(held.loc[entry["row_id"],"log_score"],entry["log_score"],rtol=2e-7,atol=1e-8)
    intervals = []
    for key,(i,row) in by_id.items():
        if row["Kind"] == "interval":
            h=held.loc[i]
            intervals.append(dict(id=key,site=row["Site"],lower_cm=row["Lower_cm"],upper_cm=row["Upper_cm"],
                                  fold=int(h["fold"]),fit_passed=h["fit_passed"],heldout_interval_probability=np.exp(h["log_score"])))
    assert len(intervals) == json.loads((REV/"provenance.json").read_text())["cases"]["principal"]["counts"]["interval"] and all(0 < r["heldout_interval_probability"] <= 1 for r in intervals)
    pd.DataFrame(intervals).to_csv(output/"interval-validation.csv",index=False)
    model = FootprintField.load(packet/"full-selected.npz")
    volumes = model.volume()
    noise = parameters(model.z)[-1]
    equal = np.full(len(volumes),1/len(volumes))
    np.testing.assert_allclose(np.quantile(volumes,[.025,.975]),receipt["conditional_interval95_km3"])
    archive = json.loads((REV/"archive-probes.json").read_text())["sites"]
    estimates = model.field(positions(archive,VENT))
    checks = []
    for j,site in enumerate(archive):
        lo,mid,hi = np.exp(np.quantile(estimates[:,j],[.025,.5,.975]))
        checks.append(dict(id=site["id"],site=site["name"],reading=site["reading"],latitude=site["latitude"],
                           longitude=site["longitude"],p025_cm=lo,median_cm=mid,p975_cm=hi,
                           prob_above_1cm=float(np.mean(estimates[:,j]>0))))
    pd.DataFrame(checks).to_csv(output/"archive-predictions.csv",index=False)

    mapped = ["D19-M04","D19-M05","D19-M11"]
    centroids = read(REV/"dull-figure9.csv")
    matched = [c for c in centroids if c["symbol"].startswith("M") and c["linked_id"].startswith("SM1-")]
    dx=np.median([float(by_id[c["linked_id"]][1]["Lon"])-float(c["longitude_from_ticks"]) for c in matched])
    dy=np.median([float(by_id[c["linked_id"]][1]["Lat"])-float(c["latitude_from_ticks"]) for c in matched])
    scenarios = [
        ("four_capped_sections",["SM1-P15","SM1-P26","SM1-P63","SM1-P66"],"SM1-P63"),
        ("uncertain_F_G_contacts",["SM1-P18","SM1-P46"],"SM1-P46"),
        ("rounded_2010_localities",[f"SM1-P{n}" for n in range(119,125)],"SM1-P119"),
        ("western_lakes",[k for k in by_id if k.startswith("D04-")],"D04-VER"),
        ("Yojoa",["SM1-P116"],"SM1-P116"),
        ("archaeological_transfers",["SH-CASA4N","I21-SAN","C95-CEREN","SM1-P67"],"I21-SAN"),
        ("Conchagua_wetland",["R10-108"],"R10-108"),
        ("figure_only_offshore",mapped,"D19-M11"),
        ("figure_registration",mapped,"D19-M11"),
        ("marine_points_and_bounds",[k for k in by_id if k in mapped or k.startswith("SM1-P") and k.split("P")[1] in {str(n) for n in range(105,115)}],"D19-M11"),
        ("La_Union_20cm_alternative",["SM1-P78"],"SM1-P78")]
    survey = [k for k in by_id if k.startswith("K09-")]
    caps = [k for k in survey if by_id[k][1]["Kind"] == "upper_limit"]
    scenarios.extend([
        ("Kitamura_field_survey",survey,"K09-46"),
        ("Kitamura_phase_ceilings",caps,caps[0]),
        ("Kitamura_eastern_sites",[k for k in survey if float(by_id[k][1]["Lon"])>-88.7],"K09-46"),
        ("Cerro_El_Carmen_wall",["E16-CARMEN"],"E16-CARMEN"),
        ("El_Trapiche_trenches",["I18-TRAP"],"I18-TRAP"),
        ("Kitamura_map_registration",survey,"K09-46")])
    checks = [p for p in read(REV/"kitamura-symbols.csv") if p["id"].startswith("K09-L")]
    survey_shift = [np.median([float(by_id[p["linked_id"]][1][field])-float(p[axis]) for p in checks])
                    for field,axis in [("Lon","longitude"),("Lat","latitude")]]
    diagnostics=[]
    for name,ids,anchor in scenarios:
        ix=[by_id[k][0] for k in ids]
        part=data.subset(np.isin(data.row_ids,ix))
        assert len(part.xy)==len(ix)
        field=model.field(part.xy)
        logw=-log_events(field,noise,part).sum(axis=1)
        shift = {"figure_registration":(dx,dy), "Kitamura_map_registration":survey_shift}.get(name)
        if shift is not None:
            source=[by_id[k][1] for k in ids]
            moved=positions([dict(longitude=float(s["Lon"])+shift[0], latitude=float(s["Lat"])+shift[1]) for s in source],VENT)
            logw+=log_events(model.field(moved),noise,part).sum(axis=1)
        if name=="La_Union_20cm_alternative":
            logw+=log_events(field,noise,replace(part,log_value=np.log([20.]))).sum(axis=1)
        weights=np.exp(logw-logsumexp(logw))
        ess=1/np.sum(weights**2)
        accepted=ess>=100 and ess/len(volumes)>=.05
        a=by_id[anchor][1]
        az=GEOD.inv(*VENT,float(a["Lon"]),float(a["Lat"]))[0]%360
        km=np.arange(0.,2001.,20.)
        lon,lat,_=GEOD.fwd(np.full(len(km),VENT[0]),np.full(len(km),VENT[1]),np.full(len(km),az),km*1000)
        line=model.field(positions([dict(longitude=x,latitude=y) for x,y in zip(lon,lat)],VENT))
        local=model.field(positions([dict(longitude=float(a["Lon"]),latitude=float(a["Lat"]))],VENT))[:,0]
        q=quantiles(volumes,weights,[.025,.5,.975]) if accepted else [None]*3
        diagnostics.append(dict(scenario=name,ids="|".join(ids),anchor=anchor,bearing_deg=az,ess=ess,
            draws=len(weights),weight_check_passed=accepted,p025_km3=q[0],median_km3=q[1],p975_km3=q[2],
            principal_local_cm=np.exp(np.median(local)),alternative_local_cm=np.exp(quantiles(local,weights,.5)) if accepted else None,
            principal_1mm_reach_km=reach(line,equal,km),alternative_1mm_reach_km=reach(line,weights,km) if accepted else None,
            delta_lon_deg=shift[0] if shift else None,delta_lat_deg=shift[1] if shift else None))
    pd.DataFrame(diagnostics).to_csv(output/"field-sensitivity.csv",index=False)

    reduced = home/"without-later-surveys"
    summary = json.loads((reduced/"fit.json").read_text())
    assert summary["engine_sha256"] == receipt["engine_sha256"]
    assert digest(reduced/"input.csv") == summary["input_sha256"]
    assert digest(reduced/"full-selected.npz") == summary["fit"]["posterior_sha256"]
    assert digest(reduced/"initial-full.npz") == summary["fit"]["attempts"][0]["posterior_sha256"]
    kept = read(reduced/"input.csv")
    assert kept == [r for r in frame if r["EvidenceID"] not in survey+["I18-TRAP","E16-CARMEN"]]
    comparison = []
    for label, fit, count, map_file, passed in [
            ("principal", model,len(frame),home/"deposit.json",receipt["fits"]["full"]["passed"]),
            ("without_later_surveys",FootprintField.load(reduced/"full-selected.npz"),len(kept),
             reduced/"map-integral.json",summary["fit"]["passed"])]:
        weights = np.full(len(fit.z),1/len(fit.z))
        qlo,q50,qhi = np.quantile(fit.volume(),[.025,.5,.975])
        numerical = json.loads(map_file.read_text())
        for site in ["SM1-P116","K09-46","E16-CARMEN"]:
            a=by_id[site][1]
            lon0,lat0=float(a["Lon"]),float(a["Lat"])
            az=GEOD.inv(*VENT,lon0,lat0)[0]%360
            km=np.arange(0.,2001.,20.)
            lon,lat,_=GEOD.fwd(np.full(len(km),VENT[0]),np.full(len(km),VENT[1]),np.full(len(km),az),km*1000)
            line=fit.field(positions([dict(longitude=x,latitude=y) for x,y in zip(lon,lat)],VENT))
            local=fit.field(positions([dict(longitude=lon0,latitude=lat0)],VENT))
            comparison.append(dict(fit=label,n=count,passed=passed,site=site,median_local_cm=np.exp(np.median(local)),
                reach_1mm_km=reach(line,weights,km),median_map_km3=numerical["median_map_volume_km3"],
                complete_draw_median_km3=q50,conditional_p025_km3=qlo,conditional_p975_km3=qhi))
    pd.DataFrame(comparison).to_csv(output/"independent-refit.csv",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=OUT)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.verify:
        with TemporaryDirectory() as temp:
            reproduce(args.folder,Path(temp))
            for item in Path(temp).glob("*.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(item),pd.read_csv(args.folder/item.name),check_dtype=False,rtol=2e-7,atol=1e-8)
    else:
        reproduce(args.folder,args.folder)
