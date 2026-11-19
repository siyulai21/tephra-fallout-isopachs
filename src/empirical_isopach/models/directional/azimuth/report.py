"""Run the uncertain-azimuth sensitivity on the same observed and withheld sites."""
from pathlib import Path
import json
import numpy as np

from ....core.catalog import dataset_record
from ..archive_readings import load_inventory
from ..ensemble.fields import Ensemble, load_members
from ..ensemble.members import splits
from ..ensemble.validation import fold_record
from ..study import digest, write_json
from ...mass_mixture.study import audited_data
from .archive import assess
from .calibration import PACKET, calibrate, case_scores, mixture_scores
from .field import AzimuthField
from .integrals import integrate, directional_reach


def model_for(data,lat,work,weights,n,kind="full",fold=None,replicate=0):
    base = Ensemble(load_members(data,lat,work,kind,fold,n=n),weights["structure"])
    return AzimuthField(base,data,weights["concentration"],replicate=replicate)


def get_scores(data,meta,work,slug,packet):
    for case in packet["cases"]:
        if case["slug"]==slug and case["input_sha256"]==meta["input_sha256"] and case["vent"]==meta["vent_lonlat"]:
            cached = {(f["kind"],f["fold"]):f for f in case["folds"]}
            compatible = True
            for kind in ("geographic","quadrants"):
                for i,train,test in splits(data,meta["vent_lonlat"][1],kind):
                    current = fold_record(work,kind,i,train,test)
                    row = cached.get((kind,i),{})
                    signatures = {name:s.get("attempts",[]) for name,s in current["members"].items()}
                    if (row.get("train")!=current["training_row_ids"] or row.get("test")!=current["withheld_row_ids"]
                            or row.get("status")!=current["status"] or row.get("member_attempts")!=signatures):
                        compatible = False
            if compatible:
                return case
    return case_scores({"slug":slug},data,meta,work)


def validation(case,weights):
    results = {}
    for kind in ("geographic","quadrants"):
        rows = []
        for f in case["folds"]:
            if f["kind"]!=kind:
                continue
            out = {k:v for k,v in f.items() if k!="log_scores"}
            if "log_scores" in f:
                before,after = mixture_scores(f,weights)
                out.update(control_score=float(-before.mean()),revised_score=float(-after.mean()),
                           control_row_log_density=before.tolist(),revised_row_log_density=after.tolist())
            rows.append(out)
        def summary(keep):
            use = [r for r in rows if "revised_score" in r and (not keep or (r["passed"] and r["phase_passed"]))]
            if not use:
                return None
            return {"folds":len(use),"rows":sum(len(r["test"]) for r in use),
                    **{k:float(np.mean([r[k] for r in use])) for k in ("control_score","revised_score")}}
        results[kind]={"all_estimated":summary(False),"passed_only":summary(True),"folds":rows}
    return results


def numerical(path,work,output,name=None,archives=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data, meta = audited_data(path)
    lat = meta["vent_lonlat"][1]
    rec = dataset_record(path) or {}
    slug = rec.get("slug","uploaded")
    packet = json.loads(PACKET.read_text())
    weights = calibrate(meta,slug,packet)
    scores = get_scores(data,meta,work,slug,packet)
    inventory,file = load_inventory(path,archives)
    attempts = []
    for n in (64,128,256,512):
        model = model_for(data,lat,work,weights,n)
        archive,w = assess(model,data,inventory,meta["vent_lonlat"])
        volume,mass = integrate(model,w)
        checks = volume["numerical"]
        attempts.append({"draws_per_chain":n,**checks})
        if checks["split_map_passed"] and checks["joint_quantile_split_passed"]:
            break
    site_check = {}
    if inventory["sites"]:
        # Independent randomization at the same sites and a denser posterior packet.
        large = model_for(data,lat,work,weights,512,replicate=1)
        site_check,_ = assess(large,data,inventory,meta["vent_lonlat"])
    members = {m.geometry.policy.name:{"draws_per_chain":len(m.index),"posterior_sha256":m.records[-1]["posterior_sha256"],
               "diagnostics":m.records[-1]["diagnostics"]} for m in model.members}
    r = {"protocol":"uncertain-azimuth-v1","dataset":name or rec.get("name",slug),"slug":slug,"input":{k:v for k,v in meta.items() if k!="transform_inv"},
         "data_rows":data.row_ids.tolist(),"calibration_sha256":digest(PACKET),"weights":weights,"members":members,
         "volume":volume,"draws_per_chain":n,"attempts":attempts,"archive_evidence_sha256":digest(file) if file else None,
         "archive":archive,"archive_independent_packet":site_check,"validation":validation(scores,weights),
         "source_connected_reach":directional_reach(model)}
    arrays = {"mass_km3":mass,"field_weights":model.weights,"concentration_index":model.k,"azimuth_rad":model.phase}
    if w is not None:
        arrays["archive_weights"] = w
    for m in model.members:
        key = m.geometry.policy.name
        arrays[key+"_indices"] = m.index
        arrays.update({key+"_"+k:v for k,v in m.draws.items()})
    np.savez_compressed(output/"fields.npz",**arrays)
    write_json(output/"report.json",r)
    return r


def audit(path,work,output):
    from ....contour_support import audit_contours
    from ....core.geometry import survey_bounds
    from ..geometry import polar, RADIUS
    output = Path(output)
    r = json.loads((output/"report.json").read_text())
    data,meta = audited_data(path)
    lat = meta["vent_lonlat"][1]
    weights,n = r["weights"],r["draws_per_chain"]
    model = model_for(data,lat,work,weights,n)
    bounds = survey_bounds(data.xy)
    corners = np.array([[x,y] for x in bounds[[0,2]] for y in bounds[[1,3]]])
    checks,deletions = [],[]
    for i,train,test in splits(data,lat,"geographic"):
        f = fold_record(work,"geographic",i,train,test)
        checks.append({k:f[k] for k in ("fold","status","passed","training_row_ids","withheld_row_ids")})
        if f["passed"]:
            deletions.append(model_for(train,lat,work,weights,n,"geographic",i))
    if polar(corners,model.members[0].geometry.axis)[0].max()>2*RADIUS:
        result = {"status":"audit_window_leaves_physical_chart","policies":{}}
    else:
        result,*_ = audit_contours(model,data,bounds,deletions)
        result["status"]="completed"
    write_json(output/"complete-contour-audit.json",{**result,"deletion_fit_status":checks,"bounds_km":bounds.tolist(),"draws_per_chain":n})
    return result

def heldout_rows(data,latitude,work,r):
    import pandas as pd
    rows = []
    for kind in ("geographic","quadrants"):
        for i,train,test in splits(data,latitude,kind):
            fold = next(f for f in r["validation"][kind]["folds"] if f["fold"]==i)
            if fold["status"]!="estimated":
                continue
            model = model_for(train,latitude,work,r["weights"],128,kind,i)
            for label,field in (("control",model.base),("revised",model)):
                predictive = field.distribution(test.xy,observation=True,log_sd=test.log_sd)
                med = np.exp(predictive.quantile(.5))
                chance = 1-predictive.cdf(np.log(10.))
                for j,row in enumerate(test.row_ids):
                    rows.append({"design":kind,"fold":i,"row_id":int(row),"method":label,
                        "fit_passed":fold["passed"],"phase_passed":fold["phase_passed"],
                        "observed_or_limit_cm":float(np.exp(test.log_value[j])),
                        "reading":"exact" if not test.censored[j] else ("lower" if test.lower_censored[j] else "upper"),
                        "exposure":fold["exposure"][j],"predicted_site_median_cm":float(med[j]),
                        "predictive_probability_ge_10cm":float(chance[j]),
                        "integrated_log_score":fold[label+"_row_log_density"][j]})
    return pd.DataFrame(rows)


def maps(path,work,output,archives=None):
    from .maps import render
    from .narrative import case_page
    output = Path(output)
    r = json.loads((output/"report.json").read_text())
    data,meta = audited_data(path)
    if meta["input_sha256"]!=r["input"]["input_sha256"]:
        raise ValueError("The requested map input differs from the fitted report.")
    inventory,_ = load_inventory(path,archives)
    lat = meta["vent_lonlat"][1]
    model = model_for(data,lat,work,r["weights"],r["draws_per_chain"])
    with np.load(output/"fields.npz") as packet:
        w = packet.get("archive_weights")
    if inventory["sites"]:
        from ..archive_readings import positions
        from .archive import phase_medians
        sites = inventory["sites"]
        medians = phase_medians(model,positions(sites,meta["vent_lonlat"]))
        r["archive"]["analytic_phase_median_cm"] = {s["id"]:v for s,v in zip(sites,medians,strict=True)}
    r["implementation_sha256"] = {p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(output/"report.json",r)
    render(output,data,meta,r,model,inventory,w)
    heldout_rows(data,lat,work,r).to_csv(output/"heldout.csv",index=False)
    case_page(output,r,inventory)
    files = ("README.md","report.json","fields.npz","complete-contour-audit.json","map-grid.npz",
             "median-isopachs.png","exceedance.png","fragment-decisions.csv","bracketed-fragments.csv","heldout.csv")
    write_json(output/"artifacts.json",{f:digest(output/f) for f in files})


def run(path,work,output,name=None,archives=None):
    result = numerical(path,work,output,name,archives)
    audit(path,work,output)
    maps(path,work,output,archives)
    return result

def attach_family_page(folder):
    folder = Path(folder)
    page = folder/"README.md"
    link = "[Measured and extrapolated directional contours](azimuth/README.md)"
    lines = page.read_text().splitlines()
    if any(link in line for line in lines):
        return
    lines[1:1] = ["",link+" shows which median contour segments the measurements bracket, how an unsampled fan changes their reach, and how the alternatives predict withheld observations."]
    page.write_text("\n".join(lines)+"\n")
    manifest = folder/"ensemble/artifacts.json"
    hashes = json.loads(manifest.read_text())
    hashes["../README.md"] = digest(page)
    write_json(manifest,hashes)
