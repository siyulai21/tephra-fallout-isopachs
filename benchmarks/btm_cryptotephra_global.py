"""Refit/replay the full-data archive response sensitivities with B-Tm excluded from calibration."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.btm_cryptotephra import AUDIT
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.study import digest, write_json
from src.empirical_isopach.models.directional.ensemble.members import run_members
from src.empirical_isopach.models.directional.azimuth.calibration import calibrate
from src.empirical_isopach.models.directional.azimuth.report import model_for
from src.empirical_isopach.models.directional.azimuth.archive import assess
from src.empirical_isopach.models.directional.azimuth.integrals import integrate
from src.empirical_isopach.models.mass_mixture.source_protocol import protocol
from src.empirical_isopach.models.mass_mixture.study import fit_checked
from src.empirical_isopach.models.mass_mixture.posterior import volume_summary
from src.empirical_isopach.models.mass_mixture.archive_check import archive_checks


def run(case, family, work, output):
    info=json.loads((AUDIT/"cryptotephra-principal.json").read_text())["cases"][case]
    path=Path(info["file"])
    data,meta=audited_data(path)
    assert meta["input_sha256"] == info["sha256"]
    output.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((AUDIT/"cryptotephra-probes.json").read_text())
    result={"case":case, "family":family, "input_sha256":info["sha256"],
            "scope":"Independent all-data refit with the same family settings. Geographic tests belong to the principal fit."}
    lat=meta["vent_lonlat"][1]
    if family=="directional":
        run_members(data,lat,work,kinds=("full",))
        weights=calibrate(meta,"b-tm")
        assert "b-tm" not in weights["calibration_cases"]
        attempts=[]
        for count in (64,128,256,512):
            model=model_for(data,lat,work,weights,count)
            volume,mass=integrate(model)
            attempts.append({"draws_per_chain":count,**volume["numerical"]})
            if volume["numerical"]["split_map_passed"] and volume["numerical"]["joint_quantile_split_passed"]:
                break
        archive,_=assess(model,data,inventory,meta["vent_lonlat"])
        result.update(weights=weights, volume=volume, refinements=attempts,
                      sites=archive["scenarios"]["quantitative"]["sites"],
                      fits={m.geometry.policy.name:m.records[-1] for m in model.members},
                      sampling_attempts={m.geometry.policy.name:m.records for m in model.members})
        packet=output/(case+"-directional-volumes.npz")
        np.savez_compressed(packet,volume_km3=mass,weights=model.weights)
    else:
        samples,fit=fit_checked(data,lat,protocol("source-fans")[0],work)
        archive,_=archive_checks(samples,data,meta["vent_lonlat"],inventory)
        result.update(volume=volume_summary(samples), fits=fit, sites=archive["quantitative"]["sites"])
        packet=output/(case+"-spherical-volumes.npz")
        np.savez_compressed(packet,volume_km3=np.exp(samples["log_volume"]))
    result["draws_sha256"]=digest(packet)
    write_json(output/(case+"-"+family+".json"),result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=["recognition_analogs","one_tenth_response"],required=True)
    p.add_argument("--family",choices=["directional","spherical"],required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(**vars(p.parse_args()))
