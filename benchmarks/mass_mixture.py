"""Whole-Earth known-truth stress tests; no contour-eligibility veto.

These fixed development cases are not proof of nominal Bayesian calibration.
The invisible cap deliberately tests a nonidentifiable scientific assumption.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from benchmarks.support import synthetic_case
from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.mass_mixture.geometry import log_field, to_sphere, RADIUS
from src.empirical_isopach.models.mass_mixture.prior import BASELINE, SENSITIVITIES
from src.empirical_isopach.models.mass_mixture.inference import Sampling
from src.empirical_isopach.models.mass_mixture.study import fit_cached


def case(scenario, seed):
    if scenario in ("smooth", "multi_lobe", "missing_sector", "missing_core"):
        data, _, volume = synthetic_case(scenario, seed)
        # Projection-centred latitude zero: equal-area radius r <= 2 R.
        # Exponential/shifted Gaussian omitted beyond the Earth chart are
        # <1e-80 of the integral. The compact proximal cap is exactly inside it.
        return data, volume
    rng = np.random.default_rng(91400+seed)
    radii, azimuth = np.meshgrid(np.linspace(40, 600, 6), np.arange(12)*np.pi/6)
    xy = np.column_stack((radii.ravel()*np.cos(azimuth.ravel()), radii.ravel()*np.sin(azimuth.ravel())))
    xy += rng.uniform(-5, 5, xy.shape)
    if scenario == "matched_two_fans":
        volume, scales, weights, offsets, exponent, eta = 120., [75., 180.], [.45, .55], [[1., 0.], [-1., 1.]], 2.5, .4
    elif scenario == "diffuse_global_tail":
        volume, scales, weights, offsets, exponent, eta = 120., [50.], [1.], [[0., 0.]], .7, 1.
    else:
        raise ValueError(scenario)
    truth = {"log_volume": np.array([np.log(volume)]), "log_scale": np.log([scales]),
             "weights": np.array([weights]), "offset": np.array([offsets]),
             "exponent": np.array([exponent]), "tail_fraction": np.array([eta])}
    value = log_field(truth, to_sphere(xy, 0))[0]+rng.normal(0, .15, len(xy))
    return Observations(xy, value, np.zeros(len(xy), bool), np.zeros(len(xy)), np.arange(len(xy))).canonical(), volume


SCENARIOS = ("smooth", "multi_lobe", "missing_sector", "missing_core", "matched_two_fans", "diffuse_global_tail")


def summarize(output, repeats=2, seed_start=10):
    output = Path(output)
    report = {"design": {"repeats": repeats, "seeds": list(range(seed_start, seed_start+repeats)), "noise_sd_log": .15,
        "benchmark_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "inherited_scenarios_sha256": sha256(Path('benchmarks/support.py').read_bytes()).hexdigest(),
        "scope": "Final development cycle, with fresh noise/location seeds after Gaussian-only and diffuse-mixture prototypes failed a smooth-thinning trial. The final model adds an exponential core and sparse Dirichlet weights; priors are fixed across scenarios and eruptions. These are development tests, not external validation or full-prior simulation-based calibration.",
        "truth_integrals": "Analytically normalized spherical fields for two added cases. Original four cases use an equal-area chart; omitted infinite-plane mass beyond 2R is <1e-80 of the volume.",
        "hidden_cap": "The added positive compact proximal cap is exactly zero at every observed site. Its volume is not identifiable from those observations.",
        "radius_km": RADIUS}, "runs": [json.loads(p.read_text()) for p in sorted((output/"runs").glob("*.json"))]}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def run(output, repeats=2, seed_start=10, scenarios=SCENARIOS):
    output = Path(output)
    (output/"runs").mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        for seed in range(seed_start, seed_start+repeats):
            data, truth = case(scenario, seed)
            # Diffuse-tail alternative is a declared sensitivity, not a fitted
            # switch selected after learning the simulated true volume.
            priors = (BASELINE, SENSITIVITIES[0]) if scenario == "diffuse_global_tail" else (BASELINE,)
            row = {"scenario": scenario, "seed": seed, "n": len(data.xy), "truth_km3": truth, "fits": {}}
            for prior in priors:
                _, result = fit_cached(data, 0, prior, output/"posteriors"/f"{scenario}-{seed}"/prior.name,
                                      Sampling(seed=20261104+seed))
                v = result["volume"]
                row["fits"][prior.name] = {**result, "covers_truth": bool(v["lower95"] <= truth <= v["upper95"]),
                                          "median_relative_error": v["median"]/truth-1}
            (output/"runs"/f"{scenario}-{seed}.json").write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
            print(scenario, seed, row["fits"]["baseline"]["volume"], flush=True)
    return summarize(output, repeats, seed_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.seed_start < 0:
        parser.error("Require positive repeats and nonnegative seeds.")
    if args.summarize_only:
        summarize(args.output, args.repeats, args.seed_start)
    else:
        run(args.output, args.repeats, args.seed_start, args.scenarios)
