"""Refit each principal global family after omitting the single lenticular western marine bed."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .sakurajima_satsuma import DATA
from src.empirical_isopach.core.audit import audited_data

CORE = "M11-MD98-2195-A"


def spherical(train, test, latitude, work):
    from src.empirical_isopach.models.mass_mixture.bearings.field import BearingSupport
    from src.empirical_isopach.models.mass_mixture.bearings.study import predictive
    from src.empirical_isopach.models.mass_mixture.bearings.volume import checked_volumes
    from src.empirical_isopach.models.mass_mixture.source_protocol import protocol
    from src.empirical_isopach.models.mass_mixture.study import fit_checked

    baseline, _ = protocol("displaced")
    samples, fit = fit_checked(train, latitude, baseline, work / "posterior")
    summary, _ = checked_volumes(samples, BearingSupport(train, latitude), initial=512, ceiling=2048)
    local, _, _ = predictive(samples, train, test, latitude)
    return dict(parent_fit=fit, principal_volume=summary, omitted_site_predictive_cm={
        key: float(np.exp(local[key][0])) for key in ("lower", "median", "upper")})


def directional(train, test, meta, work):
    from src.empirical_isopach.models.directional.azimuth.calibration import calibrate, PACKET
    from src.empirical_isopach.models.directional.azimuth.integrals import integrate
    from src.empirical_isopach.models.directional.azimuth.report import model_for
    from src.empirical_isopach.models.directional.ensemble.members import run_members

    latitude = meta["vent_lonlat"][1]
    run_members(train, latitude, work, kinds=("full",))
    weights = calibrate(meta, "sakurajima-satsuma", json.loads(PACKET.read_bytes()))
    attempts = []
    for n in (64, 128, 256, 512):
        model = model_for(train, latitude, work, weights, n)
        volume, _ = integrate(model)
        checks = volume["numerical"]
        attempts.append(dict(draws_per_chain=n, **checks))
        if checks["split_map_passed"] and checks["joint_quantile_split_passed"]:
            break
    local = model.distribution(test.xy, observation=True)
    return dict(principal_volume=volume, refinement=attempts, calibration=weights,
                fits={m.geometry.policy.name: m.records for m in model.members},
                omitted_site_predictive_cm={k: float(np.exp(local.quantile(p)[0]))
                    for k, p in (("lower", .025), ("median", .5), ("upper", .975))})



def verify_published(data, meta, sites, train):
    """Check that both published refits omit exactly the same source row and retain their diagnostics."""
    audit = Path(__file__).resolve().parents[1] / "eruptions/sakurajima-satsuma/results/source-audit"
    actual = float(np.exp(data.subset(np.isin(data.row_ids, sites)).log_value[0]))
    for family in ("directional", "spherical"):
        report = json.loads((audit / f"omit-marine-{family}.json").read_bytes())
        assert report["accepted_input_sha256"] == meta["input_sha256"]
        assert report["omitted_id"] == CORE and report["omitted_row_ids"] == sites
        assert report["retained_row_ids"] == train.row_ids.tolist()
        assert np.isclose(report["observed_cm"], actual, rtol=1e-12)
        prediction = report["omitted_site_predictive_cm"]
        assert 0 < prediction["lower"] <= prediction["median"] <= prediction["upper"] < np.inf
        volume = report["principal_volume"]
        if family == "directional":
            assert len(report["fits"]) == 6
            for attempts in report["fits"].values():
                assert isinstance(attempts[-1]["diagnostics"]["passed"], bool)
            assert volume["median_map_km3"] > 0
            quantiles = volume["coherent_volume_km3"]
            assert report["refinement"][-1]["spatial_passed"] == volume["numerical"]["spatial_passed"]
        else:
            assert isinstance(report["parent_fit"]["diagnostics"]["passed"], bool)
            assert all(isinstance(volume["checks"][k], bool) for k in ("spatial_passed", "draw_split_passed"))
            quantiles = volume["volume"]
        assert 0 < quantiles["lower95"] <= quantiles["median"] <= quantiles["upper95"] < np.inf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", nargs="?", choices=("spherical", "directional"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true", help="Check the two published omission refits.")
    args = parser.parse_args()
    data, meta = audited_data(DATA)
    with DATA.open(newline="") as stream:
        sites = [i for i, row in enumerate(csv.DictReader(stream), 2) if row["EvidenceID"] == CORE]
    if len(sites) != 1:
        raise ValueError("The omitted core must identify exactly one observed site")
    mask = np.isin(data.row_ids, sites)
    train, test = data.subset(~mask).canonical(), data.subset(mask)
    if args.verify:
        verify_published(data, meta, sites, train)
        print("Verified the retained observations, omitted core and both published refits.")
        return
    if args.output is None or args.family is None:
        parser.error("Refitting requires a family and --output.")
    work = args.output / args.family
    work.mkdir(parents=True, exist_ok=True)
    details = (spherical(train, test, meta["vent_lonlat"][1], work) if args.family == "spherical"
               else directional(train, test, meta, work))
    report = dict(omitted_id=CORE, accepted_input_sha256=meta["input_sha256"],
                  retained_row_ids=train.row_ids.tolist(), omitted_row_ids=sites,
                  observed_cm=float(np.exp(test.log_value[0])), **details)
    (work / "omission.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
