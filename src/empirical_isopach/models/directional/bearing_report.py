"""Publish directional continuation and archive-compatibility estimates for an eruption."""
from pathlib import Path
import json

import numpy as np

from ...core.catalog import dataset_record
from ..mass_mixture.study import audited_data
from .archive_readings import condition_model,load_inventory,quantitative_model,site_predictions
from .bearing_calibration import PACKET,calibrate
from .bearing_estimates import estimate,name
from .bearing_validation import screen,validate
from .ensemble.fields import Ensemble,Member
from .ensemble.members import MEMBERS
from .geometry import COMPASS,Geometry
from .bearing_support import BearingGeometry
from .study import digest,write_json


def configuration(path,work,inventory=None):
    data,meta = audited_data(path)
    data = data.canonical()
    record = dataset_record(path)
    slug = record["slug"] if record else None
    inv, source = load_inventory(path,inventory)
    weights,external = calibrate(None,slug,meta["input_sha256"],meta["vent_lonlat"])
    model = quantitative_model(data,meta["vent_lonlat"][1],work,weights)
    return data,meta,inv,source,model,external


def save_fields(folder,thin,masses):
    fields,budgets = {},{}
    members = []
    for m,a,w in zip(thin.members,masses,thin.model_weights,strict=True):
        k = name(m)
        # Quantitative-only structures with four different extrapolations share one fitted draw array.
        stored = m.records[-1]["posterior_sha256"]
        if stored+"__source" not in fields:
            for param,v in m.draws.items():
                fields[stored+"__"+param] = v
            fields[stored+"__index"] = m.index
        members.append({"name":k,"policy":m.geometry.policy.name,"range_ratio":m.geometry.ratio,
                        "array_key":stored,"weight":float(w),"chains":a.shape[0],"draws_per_chain":a.shape[1]})
        budgets[k] = a.sum(axis=-2)
    np.savez_compressed(folder/"map-fields.npz",**fields)
    np.savez_compressed(folder/"sector-draws.npz",**budgets)
    return members


def load_fields(folder,data,latitude):
    folder = Path(folder)
    r = json.loads((folder/"result.json").read_text())
    archive = folder/"map-fields.npz"
    if digest(archive) != r["artifacts"][archive.name]:
        raise ValueError("Map packet checksum mismatch")
    policies = {p.name:p for p in MEMBERS}
    members,weights = [],[]
    with np.load(archive) as saved:
        for item in r["members"]:
            key = item["array_key"]+"__"
            draws = {s[len(key):]:saved[s] for s in saved if s.startswith(key)}
            indices = draws.pop("index")
            g = BearingGeometry(Geometry(data,latitude,policies[item["policy"]]),data,item["range_ratio"])
            members.append(Member(g,draws,indices,[{"posterior_sha256":item["array_key"]}]))
            weights.append(item["weight"])
    return Ensemble(members,weights)


def record_case(path,work,output,key="quantitative",*,inventory=None,archive_work=None,mass_work=None):
    """Rebuild the posterior field after each stated reading; full estimands use actual physical area."""
    data,meta,inv,source,base,external = configuration(path,work,inventory)
    output = Path(output)
    folder = output/key
    folder.mkdir(parents=True,exist_ok=True)
    archive_work = Path(archive_work or Path(work)/"archive-refits")
    mass_work = Path(mass_work or Path(work)/"bearing-mass")
    scenario = next((s for s in inv["scenarios"] if s["id"] == key),None)
    if key!="quantitative" and scenario is None:
        raise KeyError(f"Unknown archive reading {key}")
    model,info = (base,None) if scenario is None else condition_model(
        data,base,inv["sites"],meta["vent_lonlat"],scenario,archive_work)
    sites = site_predictions(model,inv["sites"],meta["vent_lonlat"])
    r = {"protocol":"bearing-continuation-1","input_sha256":meta["input_sha256"],
         "archive_inventory_sha256":digest(source) if source else None,
         "calibration_sha256":digest(PACKET),"calibration_eruptions":external,"scenario":scenario,
         "structural_weights":model.model_weights.tolist(),"archive_likelihood":info,"sites":sites}
    if scenario is None or scenario["role"] != "cross_prediction":
        thin,complete,central,masses = estimate(model,data,mass_work/key)
        r.update(median_surface=central,coherent_fields=complete,compass_order=COMPASS,
                 map_sites=site_predictions(thin,inv["sites"],meta["vent_lonlat"]),
                 members=save_fields(folder,thin,masses))
    r["artifacts"] = {p.name:digest(p) for p in folder.glob("*.npz")}
    write_json(folder/"result.json",r)
    return r


def run(path,work,output,inventory=None):
    from .bearing_maps import render_bearing_reports
    data,meta,inv,source,base,_ = configuration(path,work,inventory)
    output = Path(output)
    scenarios = [None,*inv["scenarios"]]
    reports = {}
    for scenario in scenarios:
        key = scenario["id"] if scenario else "quantitative"
        reports[key] = record_case(path,work,output,key,inventory=source)
    for key in dict.fromkeys(("quantitative",inv.get("validation_scenario","quantitative"))):
        scenario = reports[key]["scenario"]
        details,models = validate(data,meta,work,base.model_weights,inv,scenario,
                                  Path(work)/"bearing-holdouts",count=64)
        write_json(output/key/"validation.json",details)
        surface = load_fields(output/key,data,meta["vent_lonlat"][1])
        # Compare four training deletions with a chain-balanced packet of the same size.
        from .bearing_estimates import packet
        # A saved surface packet already has its original draw index. If it was refined,
        # reconstruct the 64-per-chain full-data packet directly from the complete posterior.
        if len(surface.members[0].index) != 64:
            actual = base if scenario is None else condition_model(data,base,inv["sites"],meta["vent_lonlat"],
                                          scenario,Path(work)/"archive-refits")[0]
            surface = packet(actual,64)
        write_json(output/key/"complete-contour-audit.json",screen(data,surface,models,details))
    companion = output.parent.parent/'spherical-mixture'/'posterior.npz'
    if companion.exists() and inv['sites']:
        from ..mass_mixture.archive_check import publish
        publish(companion,data,meta,inv,output)
    render_bearing_reports(path,output,source,companion if companion.exists() else None)
    return reports


def attach_to_family_page(folder):
    folder = Path(folder)
    page = folder/"README.md"
    lines = page.read_text().splitlines()
    link = "[Directional reach and archive evidence](bearing/README.md)"
    if any(link in line for line in lines):
        return
    lines[1:1] = ["", link + " gives median isopachs, sector volumes and the sensitivity to unsampled directions. The six-structure calculation below includes a possible additional fan in the largest sampling gap."]
    page.write_text("\n".join(lines)+"\n")
    manifest = folder/"ensemble/artifacts.json"
    hashes = json.loads(manifest.read_text())
    hashes["../README.md"] = digest(page)
    write_json(manifest,hashes)
