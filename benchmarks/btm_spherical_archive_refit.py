"""Independent full-posterior check of the spherical archive-likelihood update."""
import argparse
from dataclasses import replace
from pathlib import Path

from benchmarks.btm_cryptotephra import OLD as THICKNESS_ONLY_INPUT
from src.empirical_isopach.models.directional.archive_readings import load_inventory,positions
from src.empirical_isopach.models.directional.bearing_support import include_visibility_readings
from src.empirical_isopach.models.directional.study import write_json
from src.empirical_isopach.models.mass_mixture.source_protocol import protocol
from src.empirical_isopach.models.mass_mixture.study import Sampling,audited_data,fit_checked

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    args=parser.parse_args()
    path=THICKNESS_ONLY_INPUT
    data,meta=audited_data(path)
    inv,_=load_inventory(path)
    s=next(x for x in inv["scenarios"] if x["id"]==inv["validation_scenario"])
    by_id={r["id"]:r for r in inv["sites"]}
    limits=s["effective_limit_cm"]
    xy=positions([by_id[k] for k in limits],meta["vent_lonlat"])
    obs=include_visibility_readings(data,xy,list(limits.values()))
    prior,_=protocol("source-fans")
    settings=replace(Sampling(),seed=20261231)
    _,report=fit_checked(obs,meta["vent_lonlat"][1],prior,args.work,settings)
    report.update(input_sha256=meta["input_sha256"],scenario=s)
    write_json(args.work/"archive-refit.json",report)


def publish_refit(work, output):
    import json
    import shutil
    import numpy as np
    from src.empirical_isopach.models.mass_mixture.archive_check import archive_checks
    from src.empirical_isopach.models.directional.study import digest

    work,output=Path(work),Path(output)
    path=THICKNESS_ONLY_INPUT
    data,meta=audited_data(path)
    inventory,_=load_inventory(path)
    design=json.loads((work/"archive-refit.json").read_text())
    source=work/design["posterior_file"]
    if digest(source)!=design["posterior_sha256"] or design["input_sha256"]!=meta["input_sha256"]:
        raise ValueError("Independent spherical refit SHA mismatch")
    folder=output/"spherical-refit"
    folder.mkdir(exist_ok=True)
    shutil.copyfile(source,folder/"posterior.npz")
    with np.load(source) as archive:
        fitted,_=archive_checks(dict(archive),data,meta["vent_lonlat"],{"sites":inventory["sites"],"scenarios":[]})
    reference=json.loads((output/"spherical-archive-checks.json").read_text())["scenarios"][design["scenario"]["id"]]
    report={"scenario":design["scenario"],"input_sha256":meta["input_sha256"],"volume_km3":design["volume"],
            "importance_volume_km3":reference["full_field_volume_km3"],"sites":fitted["quantitative"]["sites"],
            "sampling":design["attempts"],"diagnostics":design["diagnostics"],"source_checks":design["source_checks"],
            "posterior_sha256":design["posterior_sha256"],"normalization_refinements":design["normalization_refinements"]}
    write_json(folder/"report.json",report)
    return report
