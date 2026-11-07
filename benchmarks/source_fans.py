"""Fixed source-fan development trials, including truths that violate the constraint."""
import argparse
import json
from pathlib import Path
from hashlib import sha256
from benchmarks.mass_mixture import case, SCENARIOS
from src.empirical_isopach.models.mass_mixture.source_protocol import BASELINE, SENSITIVITIES
from src.empirical_isopach.models.mass_mixture.study import fit_checked
from src.empirical_isopach.models.mass_mixture.inference import Sampling


def run(output, scenarios=SCENARIOS, seeds=(20,21)):
    output = Path(output)
    (output/"runs").mkdir(parents=True,exist_ok=True)
    for scenario in scenarios:
        for seed in seeds:
            data,truth = case(scenario,seed)
            priors = (BASELINE,SENSITIVITIES[0]) if scenario == "diffuse_global_tail" else (BASELINE,)
            row = {"scenario":scenario,"seed":seed,"truth_km3":truth,"n":len(data.xy),"fits":{}}
            for prior in priors:
                _,fit = fit_checked(data,0,prior,output/"posteriors"/f"{scenario}-{seed}"/prior.name,
                                   Sampling(seed=20261106+seed),progress=lambda s:print(s,flush=True))
                v = fit["volume"]
                row["fits"][prior.name] = {**fit,"covers_truth":bool(v["lower95"] <= truth <= v["upper95"]),
                    "median_relative_error":v["median"]/truth-1}
            (output/"runs"/f"{scenario}-{seed}.json").write_text(json.dumps(row,indent=2,allow_nan=False)+"\n")


def summarize(output):
    output = Path(output)
    runs = [json.loads(p.read_text()) for p in sorted((output/"runs").glob("*.json"))]
    if {(r["scenario"],r["seed"]) for r in runs} != {(s,k) for s in SCENARIOS for k in (20,21)}:
        raise ValueError("Require every declared trial, including failures.")
    baseline = [r["fits"]["baseline"] for r in runs]
    passed = [f for f in baseline if f["diagnostics"]["passed"]]
    result = {"design":{"scenarios":SCENARIOS,"seeds":[20,21],"noise_sd_log":.15,
        "scope":"Development checks of a source-maximum constraint, not external validation or a 95% calibration claim. Secondary and displaced truths deliberately challenge the constraint. No fitted volume target or trial-specific priors.",
        "source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
        "truth_source_sha256":sha256(Path('benchmarks/mass_mixture.py').read_bytes()).hexdigest(),
        "inherited_truth_source_sha256":sha256(Path('benchmarks/support.py').read_bytes()).hexdigest()},
        "runs":runs,"summary":{"baseline_trials":len(baseline),"sampling_passed":len(passed),
            "intervals_containing_truth":sum(f["covers_truth"] for f in baseline),
            "passing_intervals_containing_truth":sum(f["covers_truth"] for f in passed)}}
    (output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=SCENARIOS)
    parser.add_argument('--seeds',nargs='+',type=int,choices=(20,21),default=(20,21),help='Select independent declared trials for parallel execution')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if args.summarize_only:summarize(args.output)
    else:run(args.output,args.scenarios,args.seeds)
