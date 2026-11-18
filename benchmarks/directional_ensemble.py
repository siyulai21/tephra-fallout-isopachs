"""Fit, calibrate and reproduce the six-structure directional ensemble."""
import argparse
from pathlib import Path

from benchmarks.directional import trial
from benchmarks.mass_mixture import SCENARIOS
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.directional.ensemble.members import NAMES, run_members
from src.empirical_isopach.models.directional.study import write_json
from src.empirical_isopach.models.mass_mixture.study import audited_data


def calibration(work, output):
    from src.empirical_isopach.models.directional.ensemble.validation import calibration_case
    cases = []
    for rec in datasets():
        data, meta = audited_data(rec["path"])
        cases.append(calibration_case(rec,data,meta,work/rec["slug"]))
    write_json(output, {"protocol":"buffered-eruption-stacking-v1","members":NAMES,"cases":cases})


def contour_accuracy(model, scenario, seed):
    """Area overlap for fixed thickness levels, over the entire physical Earth."""
    import numpy as np
    from scipy.stats import qmc
    from benchmarks.mass_mixture import case
    from src.empirical_isopach.models.directional.geometry import polar
    from src.empirical_isopach.models.directional.ensemble.fields import inverse_cdf, latent_arrays
    from src.empirical_isopach.models.directional.ensemble.integrals import polar_nodes
    ref = NAMES.index("reference")
    axis = model.members[ref].geometry.axis
    if scenario == "invisible_sector":
        def known(xy):
            return 5-polar(xy,axis)[0]/60
        def hidden(xy):
            d2 = np.sum((xy*[1/axis,axis]-[100,100])**2,axis=1)
            return np.log(np.exp(known(xy))+2000*np.maximum(0.,1-d2/50**2)**3)
        truths = {"sampled_equivalent":known, "hidden_lobe":hidden}
    else:
        _, _, known = case(scenario, seed, return_field=True)
        truths = {"specified":known}
    raw = {}
    for scramble in (85237, 38939):
        uv = qmc.Sobol(d=2,scramble=True,seed=scramble).random_base2(14)
        for start in range(0,len(uv),256):
            xy,area,_,_ = polar_nodes(uv[start:start+256],axis)
            arr,_ = latent_arrays(model.members,xy)
            estimates = {"ensemble_median":inverse_cdf(np.concatenate(arr).T,model.weights)[0],
                         "single_mean_log":arr[ref].mean(0)}
            for label, truth in truths.items():
                z = truth(xy)
                for method, v in estimates.items():
                    for level in (1.,10.,100.):
                        obs, pred = z>=np.log(level), v>=np.log(level)
                        union = obs|pred
                        key = (label,method,level)
                        value = raw.setdefault(key,np.zeros(4))
                        value += [area@(obs&pred),area@union,(area**2)@union,area@obs]
    result = []
    for (truth, method, level), (intersection, union, squares, known_area) in raw.items():
        result.append({"truth":truth,"method":method,"level_cm":level,
                       "true_area_km2":float(known_area/(2*2**14)),
                       "area_intersection_over_union":float(intersection/union) if union else None,
                       "union_area_ess":float(union**2/squares) if squares else 0.})
    return result


def synthetic_result(data, truth, work, output, scenario, seed):
    from src.empirical_isopach.models.directional.ensemble.report import modeled_mass
    from src.empirical_isopach.models.directional.ensemble.weights import calibrate
    # The simulated equatorial source is independent of each catalogued eruption.
    weights = calibrate({"input_sha256":"synthetic","vent_lonlat":[0.,0.]})
    model, joint, _, centre = modeled_mass(data,0.,work,weights,full=False)
    write_json(output,{"truth":truth,"median_surface":centre,"coherent_fields":joint,"weighting":weights,
                       "data_rows":data.row_ids.tolist(),
                       "contour_accuracy":contour_accuracy(model, scenario, seed)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit","calibrate","report"), default="fit")
    parser.add_argument("--output", type=Path, help="Calibration file or directory of case reports")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--synthetic", nargs="+",choices=(*SCENARIOS,"invisible_sector"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(90, 91))
    parser.add_argument("--members", nargs="+", type=int, choices=range(6), default=tuple(range(6)))
    parser.add_argument("--kinds", nargs="+", choices=("full","geographic","quadrants"),
                        default=("full","geographic","quadrants"))
    args = parser.parse_args(argv)
    if args.stage != "fit" and args.output is None:
        parser.error("Supply --output for calibration or reports")
    if args.stage == "calibrate":
        calibration(args.work,args.output)
        return
    if args.synthetic:
        for scenario in args.synthetic:
            for seed in args.seeds:
                data, truth = trial(scenario,seed)
                key = f"{scenario}-{seed}"
                work = args.work/"synthetic"/key
                if args.stage == "fit":
                    run_members(data,0.,work,("full",),args.members)
                else:
                    synthetic_result(data,truth,work,args.output/(key+".json"),scenario,seed)
        return
    from src.empirical_isopach.models.directional.ensemble.report import render
    for rec in datasets():
        if args.cases and rec["slug"] not in args.cases:
            continue
        work = args.work/rec["slug"]
        if args.stage == "report":
            render(rec["path"],work,args.output/rec["slug"],rec["name"])
        else:
            data,meta = audited_data(rec["path"])
            run_members(data,meta["vent_lonlat"][1],work,args.kinds,args.members)


if __name__ == "__main__":
    main()
