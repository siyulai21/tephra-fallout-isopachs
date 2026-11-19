"""Run independently seeded full-data visibility sensitivity fits for the B-Tm audit."""
import argparse
from pathlib import Path

from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.models.directional.archive_readings import load_inventory,positions
from src.empirical_isopach.models.directional.bearing_support import BearingGeometry,RATIOS,include_visibility_readings
from src.empirical_isopach.models.directional.ensemble.members import MEMBERS
from src.empirical_isopach.models.directional.geometry import Geometry
from src.empirical_isopach.models.directional.study import SEED,fit,write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


def fit_case(root, scenario, index):
    path = resolve_dataset("B-Tm")
    observations,meta = audited_data(path)
    data = observations.canonical()
    inventory,_ = load_inventory(path)
    sites = {s["id"]:s for s in inventory["sites"]}
    selected = scenario["effective_limit_cm"]
    xy = positions([sites[k] for k in selected],meta["vent_lonlat"])
    augmented = include_visibility_readings(data,xy,list(selected.values()))
    base = Geometry(data,meta["vent_lonlat"][1],MEMBERS[index])
    for j,ratio in enumerate(RATIOS):
        g = BearingGeometry(base,data,ratio)
        folder = root/scenario["id"]/(base.policy.name+"-"+str(ratio))
        _, info = fit(augmented,g,SEED+33000+index*100+j,folder)
        write_json(folder/"status.json",{"state":"conditional sensitivity","sites":list(selected),
                    "effective_visibility_cm":list(selected.values()),"quantitative_rows":len(data.xy),
                    "fit_geometry_uses_archives":False,**info})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--index",type=int,choices=range(3),required=True)
    parser.add_argument("--scenarios",nargs="+")
    args=parser.parse_args()
    scenarios = load_inventory(resolve_dataset("B-Tm"))[0]["scenarios"]
    requested = set(args.scenarios or (s["id"] for s in scenarios))
    if requested - {s["id"] for s in scenarios}:
        parser.error("Unknown archive scenario")
    for scenario in scenarios:
        if scenario["id"] in requested:
            fit_case(args.work,scenario,args.index)
