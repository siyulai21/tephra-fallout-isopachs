"""Common directional-continuation experiments, including an invisible-sector pair."""
import argparse
from pathlib import Path

import numpy as np

from benchmarks.mass_mixture import SCENARIOS, case
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.directional.estimate import estimate
from src.empirical_isopach.models.directional.geometry import BASE, POLICIES, RADIUS
from src.empirical_isopach.models.directional.reporting import render
from src.empirical_isopach.models.directional.study import run_fits
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.probabilistic_isopach.global_volume import EarthDomain


def trial(scenario, seed):
    if scenario != "invisible_sector":
        data, truth = case(scenario, seed)
        return data, {"total_km3": truth}
    layout, _ = case("missing_sector", seed)
    axis = EarthDomain.wgs84(0).axis_scale
    xy = layout.xy*np.array([1/axis, axis])
    r = np.linalg.norm(xy, axis=1)
    noise = np.random.default_rng(40021+seed).normal(0., .15, len(r))
    data = Observations(layout.xy, 5-r/60.+noise, layout.censored, layout.log_sd, layout.row_ids)
    # Two thickness fields coincide exactly at every site. The alternative adds
    # 2000*(1-d²/50²)^3_+ cm in a 50 km disk centred at (100,100) authalic km.
    assert np.linalg.norm(xy-[100, 100], axis=1).min() > 50.
    total = 2*np.pi*np.exp(5)*60**2*(1-(1+2*RADIUS/60)*np.exp(-2*RADIUS/60))*1e-5
    extra = np.pi*50**2*2000/4*1e-5
    return data.canonical(), {"total_km3": total, "same_data_hidden_total_km3": total+extra,
                              "NE_km3": total/8, "same_data_hidden_NE_km3": total/8+extra}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reports", type=Path, help="Also render a complete report per real eruption here")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--case", choices=[r["slug"] for r in datasets()])
    group.add_argument("--synthetic", choices=(*SCENARIOS, "invisible_sector"))
    p.add_argument("--seeds", nargs="+", type=int, default=(10, 11, 70, 71))
    args = p.parse_args(argv)
    if args.synthetic:
        for seed in args.seeds:
            data, _ = trial(args.synthetic, seed)
            work = args.output/"synthetic"/f"{args.synthetic}-{seed}"
            run_fits(data, 0, work, all_policies=False, validation=False)
            estimate(data, 0, BASE, work/"fits"/BASE.name)
    else:
        for record in datasets():
            if args.case is None or args.case == record["slug"]:
                data, meta = audited_data(record["path"])
                work = args.output/record["slug"]
                latitude = meta["vent_lonlat"][1]
                run_fits(data, latitude, work)
                if args.reports:
                    render(record["path"], work, args.reports/record["slug"])
                else:
                    for policy in POLICIES:
                        estimate(data, latitude, policy, work/"fits"/policy.name)


if __name__ == "__main__":
    main()
