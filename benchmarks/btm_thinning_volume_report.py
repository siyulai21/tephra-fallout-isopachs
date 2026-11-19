"""Report median maps and whole-deposit volumes for the B-Tm thinning-law sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from benchmarks.btm_thinning import audit_input
from benchmarks.btm_thinning_fit import VARIANTS, curve
from benchmarks.btm_thinning_mass import load_full, verify_radial_identity
from benchmarks.btm_thinning_report import LABELS
from src.empirical_isopach.models.directional.ensemble.fields import PER_CHAIN, subset_draws
from src.empirical_isopach.models.directional.ensemble.integrals import ORDERS, SEEDS, polar_nodes
from src.empirical_isopach.models.directional.geometry import polar
from src.empirical_isopach.models.directional.inference import flat
from src.empirical_isopach.models.directional.study import digest, write_json


def lower_median(values):
    """Left-inverse median for equiprobable, coherent field draws."""
    index = (len(values) - 1) // 2
    return np.partition(values, index, axis=0)[index]


def integrate_nodes(uv, geometry, params, half, radial_limits):
    inner, outer = radial_limits
    totals = np.zeros((4, 40))
    for start in range(0, len(uv), 128):
        xy, area, radius, angle = polar_nodes(uv[start:start + 128], geometry.axis)
        triangles, ids, bary = geometry.locate(xy)
        part = np.where(radius < inner, 2, np.where(radius <= outer, 3, 4))
        inside = triangles >= 0
        part[inside] = np.where(geometry.inside[triangles[inside]], 0, 1)
        groups = 8 * part + (np.floor((angle + np.pi / 8) / (np.pi / 4)).astype(int) % 8)
        values = curve(params, geometry, xy) * np.clip(1. - bary.sum(axis=1), 0., 1.)
        padded = np.pad(params["nodes"], ((0, 0), (1, 0)))
        for vertex in range(3):
            values += padded[:, ids[:, vertex] + 1] * bary[:, vertex]
        statistics = (np.exp(lower_median(values)), np.exp(lower_median(values[half])),
                      np.exp(lower_median(values[~half])), np.exp(values).mean(axis=0))
        for dest, stat in zip(totals, statistics, strict=True):
            dest += np.bincount(groups, weights=stat * area * 1e-5, minlength=40)
    return totals


def median_volume(draws, geometry, true_totals):
    radius = polar(geometry.mesh.xy, geometry.axis)[0]
    limits = radius.min(), radius.max()
    packets = []
    # Similar curves can differ by less than the Monte Carlo variation of a small map packet.
    # Retain the standard packet as a sensitivity and use every draw for the reported map.
    for count in dict.fromkeys((PER_CHAIN, draws["source"].shape[1])):
        selected, indices = subset_draws(draws, count)
        params = flat(selected)
        expected = true_totals[:, indices].mean()
        full = count == draws["source"].shape[1]
        thresholds = {"nested": .0075 if full else .015, "scramble": .01 if full else .025,
                      "mean": .005 if full else .01, "split": .025}
        half = np.tile(np.arange(len(indices)) % 2 == 0, draws["source"].shape[0])
        streams = [qmc.Sobol(2, scramble=True, seed=seed) for seed in SEEDS]
        accum = np.zeros((len(streams), 4, 40))
        previous = None
        history = []
        for power in ORDERS:
            for rep, stream in enumerate(streams):
                points = stream.random_base2(power if previous is None else power - 1)
                accum[rep] += integrate_nodes(points, geometry, params, half, limits)
            reps = accum / (2 ** power)
            avg = reps.mean(axis=0)
            central = avg[0].sum()
            spread = abs(reps[0, 0] - reps[1, 0]).sum() / central
            change = None if previous is None else abs(avg[0] - previous).sum() / central
            mean_error = abs(avg[3].sum() / expected - 1.)
            split = np.max(abs(avg[1:3].sum(axis=1) / central - 1.))
            history.append({"points_per_scramble": 2 ** power, "median_map_km3": float(central),
                            "scramble_partition_L1": float(spread), "nested_partition_L1": change,
                            "independent_mean_relative_error": float(mean_error), "draw_split_relative": float(split)})
            spatial = (change is not None and change < thresholds["nested"] and
                       spread < thresholds["scramble"] and mean_error < thresholds["mean"])
            if spatial:
                break
            previous = avg[0].copy()
        passed = bool(spatial and split < thresholds["split"])
        packets.append({"draws_per_chain": len(indices), "passed": passed, "history": history,
                        "thresholds": thresholds})
    return {"median_map_km3": float(central), "passed": passed, "packets": packets,
            "parts_sectors_km3": avg[0].reshape(5, 8).tolist(), "scramble_seeds": list(SEEDS)}


def map_variant(work, mass_dir, variant, smooth_cache):
    draws, geometry, latest = load_full(work, variant, smooth_cache)
    result = json.loads((mass_dir/(variant+".json")).read_text())
    totals_file = mass_dir/(variant+".npz")
    if digest(totals_file) != result["totals_sha256"] or latest["posterior_sha256"] != result["posterior_sha256"]:
        raise ValueError("Map and integral must use the same signed posterior")
    with np.load(totals_file) as archive:
        totals = archive["total_km3"]
    result["median_map"] = median_volume(draws, geometry, totals)
    write_json(mass_dir/(variant+"-map.json"), result)
    print(variant, result["median_map"]["median_map_km3"], result["median_map"]["passed"], flush=True)


def publish(work, mass_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    runs, all_totals, rows = {}, {}, []
    for variant in VARIANTS:
        run = json.loads((mass_dir/(variant+"-map.json")).read_text())
        path = mass_dir/(variant+".npz")
        if digest(path) != run["totals_sha256"]:
            raise ValueError(f"Volume draws changed for {variant}")
        with np.load(path) as archive:
            all_totals[variant] = archive["total_km3"]
        v = run["volume_km3"]
        rows.append({"variant": variant, "median_map_km3": run["median_map"]["median_map_km3"],
                     "coherent_median_km3": v["median"], "coherent_lower95": v["lower95"],
                     "coherent_upper95": v["upper95"], "sampler_passed": run["sampler_passed"],
                     "volume_chains_passed": run["volume_chain"]["passed"],
                     "coherent_quadrature_passed": run["quadrature"]["passed"],
                     "median_map_passed": run["median_map"]["passed"]})
        runs[variant] = run
    pd.DataFrame(rows).to_csv(output/"volumes.csv", index=False)
    np.savez_compressed(output/"volume-draws.npz", **all_totals)
    summary = ["# How much do thinning breaks change global B-Tm fallout volume?", "",
        "These six fits use the same 68 audited observations and the reference directional fan geometry. Each map integral adds the pointwise median thickness over Earth. The median and conditional interval in the next columns come from integrating each coherent deposit separately. Volumes are bulk km³. All disputed DPRK locality summaries are withheld. [Seven source-interpretation sensitivities and principal ensemble](../../README.md) · [Local predictions and held-out geographic scores](../README.md)", "",
        "| Radial thinning | Median-map integral | Whole-deposit median | Conditional 95% interval | Warnings |",
        "|---|---:|---:|---:|---|"]
    for row in rows:
        warnings = [s for s, name in (("posterior sampling", "sampler_passed"),
            ("volume chains","volume_chains_passed"),("spatial integral","coherent_quadrature_passed"),
            ("median-map integral","median_map_passed")) if not row[name]]
        summary.append(f"| {LABELS[row['variant']]} | {row['median_map_km3']:.1f} | {row['coherent_median_km3']:.1f} | {row['coherent_lower95']:.1f}–{row['coherent_upper95']:,.1f} | {', '.join(warnings) or 'None'} |")
    central = [r["median_map_km3"] for r in rows[:3]]
    summary.extend(["", f"The matched smooth, one-break and two-break median maps integrate to {min(central):.1f}–{max(central):.1f} km³. The differences are small relative to posterior uncertainty and the numerical precision of the comparison. All three share one fan geometry. The principal six-structure ensemble also represents uncertainty about the presence and width of fans in poorly sampled directions. Its 54.0 km³ median-map estimate has separate draws and diagnostics. A break in the regional thinning rate, a separate fallout fan and the local stratigraphic interpretation of a 7 m formation make different predictions.", "",
        "## Numerical integration", "",
        "Each posterior draw retains its four-chain identity, node thicknesses, fan geometry and break distances. In a radial interval with true lower radius a and slope s, the effective radius is w(a) + s(r−a). The equal-area element remains r dr dθ. Splitting background intervals at each fitted break gives an analytic integral on each piece. The footprint and its transition use positive quadrature on the same triangles and rays as the production model. The calculation extends to the antipode. Numerical checks compare all five spatial regions and eight compass sectors at successive quadrature orders. An independent adaptive scalar quadrature checks the radial formula. Setting all slope changes to unity reproduces the production integrator on complete test fields.", "",
        "The median map uses the production log-radius change of variables and two independent scrambled Sobol sequences. A preliminary map uses 64 draws per chain. The reported map and whole-deposit intervals use every draw. For the full map, the 40-part spatial integral must agree within 0.75% across nested grids and 1% across scrambles. The integrated mean field must agree within 0.5% with the independently integrated mean of the same deposits. Two interleaved posterior subsets must agree within 2.5% with the central integral. The published conditional intervals describe the complete fields. Numerical differences between closely spaced medians should be read alongside their posterior intervals and convergence diagnostics. [Complete numerical diagnostics](diagnostics.json) · [Whole-deposit draws](volume-draws.npz) · [Volume CSV](volumes.csv)", "",
        "With the matched fits saved in a fresh experiment directory, regenerate these results from the repository root.", "",
        "```sh","for v in smooth one_break two_breaks exponential broken_exponential wide_prior; do",
        '  python -m benchmarks.btm_thinning_mass --work results/btm-break-check --output results/btm-break-mass --variant "$v"',
        '  python -m benchmarks.btm_thinning_volume_report --work results/btm-break-check --mass results/btm-break-mass --variant "$v"',
        "done","python -m benchmarks.btm_thinning_volume_report --work results/btm-break-check --mass results/btm-break-mass --output results/btm-break-report/volumes","```", ""])
    (output/"README.md").write_text("\n".join(summary))
    here = Path(__file__).resolve().parent
    write_json(output/"diagnostics.json", {"runs": runs,
        "code_sha256": {name: digest(here/name) for name in ("btm_thinning_mass.py","btm_thinning_volume_report.py")},
        "files": {p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name != "diagnostics.json"}})
    verify(output, work/"smooth/result.json")


def verify(folder, smooth_fit=None):
    verify_radial_identity()
    info = json.loads((folder/"diagnostics.json").read_text())
    here = Path(__file__).resolve().parent
    for name, sha in info["code_sha256"].items():
        assert digest(here/name) == sha
    for name, sha in info["files"].items():
        assert digest(folder/name) == sha
    fits = (json.loads(smooth_fit.read_text())["full_fit"] if smooth_fit
            else json.loads((folder.parent/"fits.json").read_text())["smooth"]["full_fit"])
    assert info["runs"]["smooth"]["posterior_sha256"] == fits["attempts"][-1]["posterior_sha256"]
    summary = pd.read_csv(folder/"volumes.csv").set_index("variant")
    with np.load(folder/"volume-draws.npz") as archive:
        assert set(archive.files) == set(summary.index) == set(info["runs"]) == set(VARIANTS)
        for key, run in info["runs"].items():
            a = archive[key]
            assert list(a.shape) == run["volume_draw_shape"] and a.shape[0] == 4
            assert np.isfinite(a).all() and np.all(a > 0)
            assert run["input_sha256"] == digest(audit_input())
            np.testing.assert_allclose(np.quantile(a,[.025,.5,.975]), summary.loc[key,
                ["coherent_lower95","coherent_median_km3","coherent_upper95"]].astype(float))
            map_result = run["median_map"]
            np.testing.assert_allclose(np.sum(map_result["parts_sectors_km3"]), summary.at[key,"median_map_km3"])
            end = map_result["packets"][-1]["history"][-1]
            assert map_result["packets"][-1]["draws_per_chain"] == a.shape[1]
            assert map_result["passed"] == (end["nested_partition_L1"] is not None and
                end["nested_partition_L1"] < .0075 and end["scramble_partition_L1"] < .01 and
                end["independent_mean_relative_error"] < .005 and end["draw_split_relative"] < .025)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--mass", type=Path, required=True)
    parser.add_argument("--smooth-cache", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--variant", choices=VARIANTS)
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.variant:
        map_variant(args.work,args.mass,args.variant,args.smooth_cache)
    else:
        publish(args.work,args.mass,args.output)
