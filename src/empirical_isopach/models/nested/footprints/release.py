"""Fit, check and publish the nested footprint model independently of corridor fits."""
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ....core.audit import audited_data
from ....core.catalog import dataset_record
from ....visualization.deposit import render_deposit
from ...directional.ensemble.members import splits
from ...extent_support import save_screen
from ..study import digest, write_json
from . import diagnostics
from .posterior import FootprintField, PROTOCOL, write_fit


def fit_part(folder, key, train, test, meta):
    initial = write_fit(folder,key,train,test,meta)
    if initial["numerics"]["passed"]:
        return initial
    # Inspect the continued-volume functional too. Every unsuccessful production
    # partition receives the same larger four-population attempt.
    dest = folder/"refinement"
    return write_fit(dest,key,train,test,meta,initial=12288,ceiling=12288)


def sector_checks(folder, data, model, report):
    from .sectors import budgets
    frame, checks = budgets(model,data,report["map_draws"])
    folder=Path(folder)/"footprints"
    frame.to_csv(folder/"sector-volumes.csv",index=False,lineterminator="\n")
    write_json(folder/"sector-checks.json",{**checks,"input_sha256":report["input_sha256"],
               "posterior_sha256":report["posterior_sha256"]["full-selected.npz"],"table_sha256":digest(folder/"sector-volumes.csv")})


def describe_geography(folder, report):
    folder = Path(folder)
    packet = folder/"footprints"
    sectors = pd.read_csv(packet/"sector-volumes.csv")
    counts = sectors[["exact","lower","upper","interval"]].sum(axis=1)
    empty = sectors[counts==0]
    fraction = report["sensitivities"]["rank_partitions"]
    first, _, last = fraction["median_mass_fractions"]
    lo, hi = fraction["outside_last_positive_fraction95"]
    with gzip.open(packet/"support.json.gz","rt") as stream:
        audit = json.load(stream)
    policies = audit.get("policies",{})
    accepted = sum(bool(p.get("selected_band")) for p in policies.values())
    lines = ["## Geographic support and extrapolation", "",
             f"Across draws, the median volume share inside the first measured area rank is {first:.1%}. "
             f"The median share outside the last positive rank is {last:.1%}, with a conditional 95% range of {lo:.1%}–{hi:.1%}. "
             "A measured site can constrain one part of a perimeter while its opposite side remains unsampled.", ""]
    if policies:
        lines.append(f"The shared geographic screen accepts a complete contour band under {accepted} of {len(policies)} policies.")
    else:
        lines.append(f"The shared geographic screen reports {audit['status'].replace('_',' ')}.")
    if len(empty):
        subject = "sector contains" if len(empty)==1 else "sectors contain"
        lines[-1] += (f" The {', '.join(empty.sector)} {subject} no accepted off-vent sites. "
                      f"These bearings account for {empty.median_map_fraction.sum():.1%} of the median-map volume.")
    lines += ["","[Sector-by-sector evidence and volume](footprints/sector-volumes.csv) · "
              "[Volume sensitivity to the priors](footprints/report.json) · [Complete contour decisions](footprints/support.json.gz)",""]
    page = folder/"README.md"
    body = page.read_text().split("\n## Geographic support and extrapolation")[0].rstrip()
    page.write_text(body+"\n\n"+"\n".join(lines))


def archive_checks(path, folder, data, meta, model, archives=None):
    from ..report import archive_predictions
    probes = archive_predictions(path, archives, data, meta, model)
    write_json(Path(folder)/"footprints/archive-probes.json",
               {"input_sha256":meta["input_sha256"],"posterior_sha256":digest(Path(folder)/"footprints/full-selected.npz"),**probes})


def assemble(path, folder, name, *, upstream=None, archives=None):
    """Accept verified run packets or fit all folds; preserve every earlier numerical attempt."""
    data, meta = audited_data(path)
    fits = Path(folder)/"footprints"
    fits.mkdir(parents=True,exist_ok=True)
    source = Path(upstream) if upstream else Path(folder)/"work"/"footprint-fits"
    models, hashes, receipts, predictions = [], {}, {}, []
    for i,train,test in [(None,data,None),*splits(data,meta["vent_lonlat"][1],"geographic")]:
        key = "full" if i is None else f"fold-{i}"
        rec = fit_part(source,key,train,test,meta)
        start = json.loads((source/(key+".json")).read_text())
        refined = not start["numerics"]["passed"]
        origin = source/"refinement" if refined else source
        packet = origin/(key+".npz")
        # Presentation packets are distinct from scratch so both raw fit attempts stay reproducible.
        target = fits/(key+"-selected.npz")
        if digest(packet)!=rec["posterior_sha256"]:
            raise ValueError(f"Selected posterior changed at {packet}")
        if not target.exists() or digest(target)!=rec["posterior_sha256"]:
            shutil.copyfile(packet,target)
        hashes[target.name] = rec["posterior_sha256"]
        history = [{"identity":start["identity"],"posterior_sha256":start["posterior_sha256"],"numerics":start["numerics"]}]
        if refined:
            history.append({"identity":rec["identity"],"posterior_sha256":rec["posterior_sha256"],"numerics":rec["numerics"]})
        receipts[key] = {"posterior_sha256":rec["posterior_sha256"],"attempts":history,"passed":rec["numerics"]["passed"],
                         "train":rec["identity"]["train"],"test":rec["identity"]["test"],
                         "draws":rec["draws"],"volume_quantiles_km3":rec["volume_quantiles_km3"]}
        m = FootprintField.load(target)
        models.append((m,rec["numerics"]["passed"]))
        if i is None:
            full, info = m, rec
        else:
            predictions.extend(dict(fold=i,fit_passed=rec["numerics"]["passed"],**s) for s in rec["heldout"])
    frame = pd.DataFrame(predictions)
    frame.to_csv(fits/"heldout.csv.gz",index=False,compression={"method":"gzip","mtime":0},lineterminator="\n")
    scores = -frame.groupby("fold").log_score.mean()
    exact = frame[frame.exact]
    validation = dict(negative_log_score=float(scores.mean()),fold_scores=scores.tolist(),
                      all_folds_passed=all(ok for _,ok in models[1:]),passed_folds=sum(ok for _,ok in models[1:]),
                      coverage95_exact=float(exact.cdf.between(.025,.975).mean()) if len(exact) else None,
                      scored_rows=len(frame),exact_rows=len(exact))
    histories = diagnostics.integral_check(full)
    maps = diagnostics.map_draws(full)
    save_screen(fits/"support.json.gz",diagnostics.support(data,meta,full,models[1:]))
    report = dict(protocol=PROTOCOL, family="nested-isopachs",name=name,input_sha256=meta["input_sha256"],
                  accepted_rows=len(data.xy),source_latitude=full.latitude,posterior_sha256=hashes,
                  engine_sha256=rec["identity"]["engine_sha256"],fits=receipts,
                  conditional_interval95_km3=info["volume_quantiles_km3"][::2],
                  complete_draw_median_km3=info["volume_quantiles_km3"][1],
                  draw_checks_passed=bool(info["numerics"]["passed"] and histories[-1]["passed"]),
                  validation=validation,posterior_family=info["posterior_family"],sensitivities=diagnostics.sensitivity(full,data),
                  independent_draw_integration=histories,median_draw_refinements=maps,
                  map_draws=maps[-1]["draws"],heldout_sha256=digest(fits/"heldout.csv.gz"),support_sha256=digest(fits/"support.json.gz"))
    sector_checks(folder,data,full,report)
    archive_checks(path,folder,data,meta,full,archives)
    write_json(fits/"report.json",report)
    return data,meta,report


def run(path, output, name=None, *, resume=False, upstream=None, archives=None):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise ValueError("Use a fresh nested result folder or resume a verified run.")
    published = folder/"footprints/report.json"
    if resume and published.is_file():
        identity = json.loads(published.read_text())
        if identity.get("input_sha256") != digest(path) or identity.get("protocol") != PROTOCOL:
            raise ValueError(f"Resume inputs or model differ from {published}")
    if resume and (folder/"full.json").is_file():
        existing = json.loads((folder/"full.json").read_text()).get("identity", {})
        if existing.get("input_sha256") != digest(path):
            raise ValueError(f"Resume inputs or model differ from {folder}/full.json")
    name = name or (dataset_record(path) or {}).get("name",Path(path).stem)
    data,meta,report = assemble(path,folder,name,upstream=upstream,archives=archives)
    root_report = folder/"report.json"
    if not root_report.exists() or json.loads(root_report.read_text())["protocol"] == PROTOCOL:
        write_json(root_report,report)
    if not (folder/"README.md").exists():
        (folder/"README.md").write_text(f"# {name} · nested isopachs\n")
    mapped = render_deposit(folder,"nested-isopachs",data,meta,name)
    describe_geography(folder,report)
    return {**report, "median_map_volume_km3":mapped["median_map_volume_km3"]}
