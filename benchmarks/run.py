"""Run: python -m benchmarks.run --output results/benchmark

The synthetic test surfaces are not generated from the fitted GP, to expose
misspecification. This small benchmark is diagnostic, not external validation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.probabilistic_isopach import Observations, SpatialModel, Candidate, read_observations
from src.probabilistic_isopach.validation import select_model, training_mask
from src.probabilistic_isopach.regularized_spline import select_spline_prior
from src.probabilistic_isopach.__main__ import main as fit_cli
from src.spline_isopach.lsf_bsplines2d import LsfBsplines2d


def real_baselines(data, report):
    """Same outer folds as nested validation. No manual pseudo data in any fit."""
    folds = np.array(report["nested_validation"]["folds"])
    d = data.canonical()
    radial, spline = np.empty(len(d.xy)), np.empty(len(d.xy))
    for fold in range(4):
        train, test = training_mask(d, folds, fold, 0.)
        tr = d.subset(train)
        # A fixed-family trend baseline, not tuned using held-out points.
        model = SpatialModel(Candidate("exponential")).fit(tr)
        radial[test] = model._features(d.xy[test])@model.beta
        old = LsfBsplines2d(tr.xy[:, 0], tr.xy[:, 1], np.exp(tr.log_value),
                           np.zeros(len(tr.xy)), .25, 3.2, {})
        old.fit()
        spline[test] = [old.eval_cubic_bspline_surface(*xy) for xy in d.xy[test]]
    return {"trend_only_rmse_log": float(np.sqrt(np.mean((radial-d.log_value)**2))),
            "corrected_spline_no_pseudo_rmse_log": float(np.sqrt(np.mean((spline-d.log_value)**2))),
            "note": "Fixed exponential trend and corrected spline at legacy defaults; spline clips predictions to training rectangle. These baselines are not independently tuned."}


def synthetic_benchmark(repeats):
    rows = []
    for scenario in ("smooth", "secondary_lobe", "missing_sector"):
        for seed in range(repeats):
            rng = np.random.default_rng(seed+200)
            xy = rng.uniform(-250, 250, (120, 2))
            if scenario == "missing_sector":
                xy = xy[~((xy[:, 0] > 0) & (xy[:, 1] > 0))]
            xy = xy[:60]
            test = rng.uniform(-250, 250, (400, 2))
            def truth(p):
                r = np.linalg.norm(p, axis=1)
                base = 5-.018*r+.005*p[:, 0]
                if scenario != "smooth":
                    base += 2.5*np.exp(-((p[:, 0]-130)**2+(p[:, 1]-100)**2)/(2*45**2))
                return base
            latent = truth(xy)
            observed = latent+rng.normal(0, .25, len(xy))
            data = Observations(xy, observed, np.zeros(len(xy), bool), np.zeros(len(xy)), np.arange(len(xy)))
            model, _ = select_model(data)
            mean, var = model.predict_log(test)
            test_latent = truth(test)
            heldout_y = test_latent+rng.normal(0, .25, len(test))
            obs_var = var+model.noise**2
            domain = (500*500)*1e-5
            estimated = np.exp(mean+var/2).mean()*domain
            true_volume = np.exp(test_latent).mean()*domain
            rows.append({"scenario": scenario, "seed": seed, "selected_tail": model.candidate.tail,
                         "latent_rmse_log": float(np.sqrt(np.mean((mean-test_latent)**2))),
                         "latent_coverage95": float(np.mean(abs(mean-test_latent) <= 1.95996398454*np.sqrt(var))),
                         "observation_coverage95": float(np.mean(abs(mean-heldout_y) <= 1.95996398454*np.sqrt(obs_var))),
                         "volume_relative_error_matched_mc_grid": float((estimated-true_volume)/true_volume)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    real = []
    for name in ("Mazama", "Tambora", "Aso-4", "B-Tm"):
        csv = Path(f"examples/Data/{name}_tephra.csv")
        dest = args.output/name
        fit_cli([str(csv), "--output", str(dest), "--nested-cv"])
        report = json.loads((dest/"report.json").read_text())
        data, _ = read_observations(csv)
        prior, prior_records = select_spline_prior(data)
        validation = report["nested_validation"]
        real.append({"dataset": name, "n_real": len(data.xy),
                     "ignored_pseudo": len(report["input"]["ignored_pseudo_rows"]),
                     "selected": report["model"]["candidate"],
                     "nested_rmse_log": validation["rmse_log"],
                     "nested_coverage95": validation["coverage95_observations"],
                     "nested_log_score": validation["mean_log_score"],
                     "baselines": real_baselines(data, report), "warnings": report["warnings"],
                     "spline_prior": {"selected_strength": prior.strength,
                                      "selection_diagnostics": prior_records,
                                      "total_precision": prior.prior_total_precision},
                     "volume": report["volume"], "domain_sensitivity": report["domain_sensitivity"],
                     "grid_check": report["volume_grid_check"]})
        print(f"Finished {name}", flush=True)
    synthetic = synthetic_benchmark(args.repeats)
    result = {"real": real, "synthetic": synthetic,
              "design": {"real_folds": 4, "inner_folds": 3, "buffer_km": 0,
                         "synthetic_repeats": args.repeats, "synthetic_observation_noise_log": .25,
                         "synthetic_test_sites_per_run": 400,
                         "interpretation": "Diagnostic benchmark; conditional intervals and finite-domain volumes, not validated eruption volumes."}}
    (args.output/"benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Saved {args.output/'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
