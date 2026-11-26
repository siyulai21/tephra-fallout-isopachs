"""Fit, check and publish the nested footprint model independently of corridor fits."""
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ....core.fitting import model_input
from ....core.audit import audited_data
from ....core.catalog import dataset_record
from ....visualization.deposit import render_deposit
from ...directional.ensemble.members import splits
from ...extent_support import save_screen
from ..study import digest, write_json
from . import diagnostics
from . import map_convergence
from . import rejuvenation
from .continuous import FlowField as FootprintField, PROTOCOL, write_fit


def fit_part(folder, key, train, test, meta):
    folder=Path(folder)
    map_checks=[]
    quantile_checks=[]
    def checked(parent,receipt):
        if key!="full":
            return receipt
        packet=parent/(key+".npz")
        path=parent/(key+"-field-convergence.json")
        identity=dict(input_sha256=meta["input_sha256"],posterior_sha256=receipt["posterior_sha256"],
                      diagnostic_sha256=digest(diagnostics.__file__))
        if path.exists():
            saved=json.loads(path.read_text())
            if saved["identity"]!=identity:
                raise ValueError(f"Spatial convergence identity differs at {path}")
            check=saved["check"]
        else:
            check=diagnostics.population_field_check(FootprintField.load(packet),train)
            write_json(path,dict(identity=identity,check=check))
        map_checks.append(dict(identity=identity,check=check))
        quantiles_passed=True
        if receipt['numerics']['passed'] and check['passed']:
            path=parent/(key+'-mapped-quantiles.json')
            quantile_identity={**identity,'diagnostic_sha256':digest(map_convergence.__file__)}
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['identity']!=quantile_identity:
                    raise ValueError(f'Mapped quantile convergence identity differs at {path}')
                trials=saved['checks']
            else:
                trials=map_convergence.map_draws(FootprintField.load(packet))
                write_json(path,dict(identity=quantile_identity,checks=trials))
            quantile_checks.append(dict(identity=quantile_identity,checks=trials))
            quantiles_passed=trials[-1]['passed']
        return {**receipt,"raw_fit_passed":receipt["numerics"]["passed"],"field_population_checks":list(map_checks),
                'mapped_quantile_checks':list(quantile_checks),
                "numerics":{**receipt["numerics"],"passed":bool(receipt["numerics"]["passed"] and check["passed"] and quantiles_passed)}}
    initial = checked(folder,write_fit(folder,key,train,test,meta))
    if initial["numerics"]["passed"]:
        return initial
    # Inspect the continued-volume functional too. Every unsuccessful production
    # partition receives the same larger four-population attempt.
    for particles in (12288,24576):
        dest = folder/("refinement" if particles==12288 else "refinement-24576")
        result = checked(dest,write_fit(dest,key,train,test,meta,initial=particles,ceiling=particles))
        if result["numerics"]["passed"]:
            return result
    dest=folder/'rejuvenation'
    result=rejuvenation.write_fit(dest,key,train,test,meta,
                                  folder/'refinement-24576'/(key+'.npz'),result)
    return checked(dest,result)


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
        selected_particles = rec["identity"]["initial"]
        moved = rec['identity'].get('sampler')==rejuvenation.SAMPLER
        refined = selected_particles>=12288 or moved
        origin = source/'rejuvenation' if moved else (source/("refinement" if selected_particles==12288 else "refinement-24576")) if refined else source
        packet = origin/(key+".npz")
        # Presentation packets are distinct from scratch so both raw fit attempts stay reproducible.
        target = fits/(key+"-selected.npz")
        if digest(packet)!=rec["posterior_sha256"]:
            raise ValueError(f"Selected posterior changed at {packet}")
        if not target.exists() or digest(target)!=rec["posterior_sha256"]:
            shutil.copyfile(packet,target)
        hashes[target.name] = rec["posterior_sha256"]
        history = [{"identity":start["identity"],"posterior_sha256":start["posterior_sha256"],"numerics":start["numerics"]}]
        attempts=[(source,start)]
        if refined:
            for particles in (12288,24576):
                if not moved and particles>selected_particles: break
                parent=source/("refinement" if particles==12288 else "refinement-24576")
                attempt=rec if parent==origin else json.loads((parent/(key+".json")).read_text())
                attempts.append((parent,attempt))
                history.append({"identity":attempt["identity"],"posterior_sha256":attempt["posterior_sha256"],"numerics":attempt["numerics"]})
        if moved:
            attempts.append((origin,rec))
            history.append({'identity':rec['identity'],'posterior_sha256':rec['posterior_sha256'],'numerics':rec['numerics']})
        checkpoint_records=[]
        for parent,attempt in attempts:
            for checkpoint in attempt.get("checkpoints",[]):
                relative=parent.relative_to(source)/checkpoint["packet"]
                if digest(parent/checkpoint["packet"])!=checkpoint["posterior_sha256"]:
                    raise ValueError(f"Refinement posterior differs at {parent/checkpoint['packet']}")
                if checkpoint["posterior_sha256"]==rec["posterior_sha256"]:
                    # The selected field is also the final checkpoint. Keep
                    # one canonical packet while preserving both identities.
                    relative=Path(target.name)
                else:
                    dest=fits/relative
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(parent/checkpoint["packet"],dest)
                checkpoint_records.append({**checkpoint,"packet":relative.as_posix(),"identity":attempt["identity"]})
        receipts[key] = {"posterior_sha256":rec["posterior_sha256"],"attempts":history,"passed":rec["numerics"]["passed"],
                         "train":rec["identity"]["train"],"test":rec["identity"]["test"],
                         "draws":rec["draws"],"volume_quantiles_km3":rec["volume_quantiles_km3"],
                         "checkpoints":checkpoint_records,
                         "field_population_checks":rec.get("field_population_checks",[]),
                         'mapped_quantile_checks':rec.get('mapped_quantile_checks',[])}
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
    checked=info.get('mapped_quantile_checks',[])
    maps = (checked[-1]['checks'] if checked and checked[-1]['identity']['posterior_sha256']==info['posterior_sha256']
            else map_convergence.map_draws(full))
    save_screen(fits/"support.json.gz",diagnostics.support(data,meta,full,models[1:]))
    report = dict(protocol=PROTOCOL, family="nested-isopachs",name=name,input_sha256=meta["input_sha256"],
                  accepted_rows=len(data.xy),source_latitude=full.latitude,posterior_sha256=hashes,
                  engine_sha256=rec["identity"]["engine_sha256"],fits=receipts,
                  conditional_interval95_km3=info["volume_quantiles_km3"][::2],
                  complete_draw_median_km3=info["volume_quantiles_km3"][1],
                  draw_checks_passed=bool(info["numerics"]["passed"] and histories[-1]["passed"]),
                  validation=validation,posterior_family=info["posterior_family"],sensitivities=diagnostics.sensitivity(full,data),
                  independent_draw_integration=histories,median_draw_refinements=maps,
                  map_draw_diagnostic_sha256=digest(map_convergence.__file__),
                  map_draws=maps[-1]["draws"],heldout_sha256=digest(fits/"heldout.csv.gz"),support_sha256=digest(fits/"support.json.gz"))
    sector_checks(folder,data,full,report)
    archive_checks(path,folder,data,meta,full,archives)
    write_json(fits/"report.json",report)
    return data,meta,report


def run(path, output, name=None, *, resume=False, upstream=None, archives=None):
    path = model_input(path, output)
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
