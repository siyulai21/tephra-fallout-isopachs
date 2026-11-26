"""Frozen-policy recovery of known deposits, repeated surveys and independent field campaigns."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .footprint_challenges import rotated_spokes, synthetic_id
from .nested_reach_challenge import known_design
from .nested_reach_field import partition, calbuco
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.directional.geometry import polar
from src.empirical_isopach.models.directional.multiscale.fit import fit
from src.empirical_isopach.models.directional.multiscale.validation import predictions, write_json
from src.empirical_isopach.models.nested.reach_integrals import nodes
from src.probabilistic_isopach.global_volume import EarthDomain

KNOWN = (
    "smooth",
    "multi_lobe",
    "matched_two_fans",
    "missing_sector",
    "invisible_sector",
    "broad_umbrella",
    "survey_rotation_0",
    "survey_rotation_23",
    "turning_fans",
)
SURVEYS = (
    "calbuco-2015",
    "aso-4",
    "b-tm",
    "pinatubo-1991-layer-c",
    "mazama",
    "ilopango-tbj",
    "pinatubo-interpreted",
    "sihailongwan",
)


def checked_fit(data, latitude, out):
    attempts = []
    for power in (8, 10, 11):
        model, receipt = fit(data, latitude, power=power)
        attempts.append(dict(power=power, **receipt))
        if receipt["passed"]:
            break
    write_json(out / "fit-attempts.json", attempts)
    return model, receipt


def select_engine(previous=False):
    """Run the preserved v5 engine on today's identical synthetic observations."""
    if not previous:
        return
    import importlib
    import sys
    import types

    name = "src.empirical_isopach.models.directional._v5_control"
    package = types.ModuleType(name)
    package.__path__ = [str(Path("archive/2026-11-25-directional-v5").resolve())]
    sys.modules[name] = package
    global fit
    fit = importlib.import_module(name + ".fit").fit


def turning(seed):
    axis = EarthDomain.wgs84(0.0).axis_scale

    def truth(xy):
        r, t = polar(xy, axis)
        phase = 1.4 * r * r / (80.0**2 + r * r)
        gain = 35.0 + 140 * np.exp(3.0 * (np.cos(t - phase) - 1)) + 82 * np.exp(5.0 * (np.cos(t + 1.7) - 1))
        return np.log(180.0) - r / gain

    rng = np.random.default_rng(67703 + seed)
    r = np.exp(rng.uniform(np.log(3.0), np.log(450.0), 90))
    t = rng.uniform(0.0, 2 * np.pi, len(r))
    xy = np.column_stack((r * np.sin(t) * axis, r * np.cos(t) / axis))
    y = truth(xy) + rng.normal(0.0, 0.22, len(r))
    data = Observations(xy, y, np.zeros(len(r), bool), np.zeros(len(r)), np.arange(len(r))).canonical()
    pts, w, *_ = nodes(18, 85613, 0.0)
    return data, {"specified": (truth, float(1e-5 * np.mean(w * np.exp(truth(pts)))))}


def known(design, seed, out):
    if design.startswith("survey_rotation"):
        data, truths = rotated_spokes(design)
    elif design == "turning_fans":
        data, truths = turning(seed)
    else:
        data, truths = known_design(design, seed)
    model, fit_info = checked_fit(data, 0.0, out)
    rows = []
    for scramble in (51071, 68389):
        xy, w, *_ = nodes(13, scramble, 0.0)
        w /= len(w)
        z = model.central(xy) * np.log(10.0)
        for label, (truth, volume) in truths.items():
            actual = truth(xy)
            for cm in (0.5, 1.0, 10.0, 100.0):
                observed, pred = actual >= np.log(cm), z >= np.log(cm)
                union = w[observed | pred].sum()
                rows.append(
                    dict(
                        design=design,
                        seed=seed,
                        scramble=scramble,
                        truth=label,
                        input_sha256=synthetic_id(data),
                        truth_volume_km3=volume,
                        method="multiscale",
                        median_map_km3=float(w @ np.exp(z) * 1e-5),
                        level_cm=cm,
                        false_area_km2=float(w[pred & ~observed].sum()),
                        missed_area_km2=float(w[observed & ~pred].sum()),
                        intersection_over_union=float(w[observed & pred].sum() / union) if union else None,
                        fit_passed=fit_info["passed"],
                    )
                )
    pd.DataFrame(rows).to_csv(out / "recovery.csv", index=False)
    from src.empirical_isopach.models.directional.multiscale.integrals import integrate

    sampled = integrate(model, data, out / "volume-draws.npz")
    lo, hi = sampled["conditional_interval95_km3"]
    totals = [
        dict(
            design=design,
            seed=seed,
            truth=label,
            input_sha256=synthetic_id(data),
            truth_volume_km3=volume,
            lower95_km3=lo,
            upper95_km3=hi,
            covers_truth=lo <= volume <= hi,
            fit_passed=fit_info["passed"],
            integration_passed=sampled["integration_passed"],
        )
        for label, (_, volume) in truths.items()
    ]
    pd.DataFrame(totals).to_csv(out / "volume-intervals.csv", index=False)
    write_json(out / "volume-refinements.json", sampled["independent_draw_integration"])
    if design.startswith("survey_rotation"):
        a = np.arange(360) * np.pi / 180
        r = np.geomspace(4.0, 800.0, 193)
        xy = np.stack(
            (
                model.domain.axis_scale * np.outer(r, np.sin(a)),
                np.outer(r, np.cos(a)) / model.domain.axis_scale,
            ),
            axis=-1,
        ).reshape(-1, 2)
        z = model.central(xy).reshape(len(r), len(a))
        from src.empirical_isopach.visualization.deposit_field import PolarField

        f = PolarField(r, z, model.domain.axis_scale)
        pd.DataFrame({str(cm): f.outer_crossings(cm) for cm in (0.5, 1.0, 10.0)}).to_csv(
            out / "radii.csv", index=False
        )


def campaign(slug, out):
    if slug in ("aso-4", "b-tm"):
        train, test, meta, _ = partition(slug, out)
    elif slug == "calbuco-2015":
        train, test, meta, _ = calbuco(out)
    else:
        rec = next(
            r
            for r in datasets()
            if r["slug"]
            == ({"pinatubo-interpreted": "pinatubo-1991-layer-c", "sihailongwan": "b-tm"}.get(slug, slug))
        )
        data, meta = audited_data(rec["path"])
        if slug == "sihailongwan":
            select = np.isin(
                data.row_ids,
                [v["row_id"] for v in meta["interpreted_upper_limits"] if v["id"] == "longgang"],
            )
        elif slug in ("mazama", "ilopango-tbj", "pinatubo-interpreted"):
            select = np.isin(data.row_ids, [v["row_id"] for v in meta["interpreted_upper_limits"]])
        else:
            # Entire marine collection in the current 235-row edition, cross-cruise duplicates already audited.
            ids = pd.read_csv(rec["path"]).iloc[data.row_ids - 2].EvidenceID.str.split("-").str[0]
            select = ids.isin(("H01", "WW", "W04", "SO95", "W09", "S99", "W08")).to_numpy()
        train, test = data.subset(~select), data.subset(select)
    model, receipt = checked_fit(train, meta["vent_lonlat"][1], out)
    rows = predictions(model, test, "survey", 0, receipt["passed"])
    pd.DataFrame(rows).assign(
        survey=slug, input_sha256=meta["input_sha256"], training_rows=len(train.xy)
    ).to_csv(out / "survey.csv", index=False)
    volumes = []
    for s in (51071, 68389):
        xy, w, *_ = nodes(13, s, meta["vent_lonlat"][1])
        volumes.append(float(1e-5 * np.mean(w * 10.0 ** model.central(xy))))
    write_json(
        out / "survey-fit.json",
        dict(
            survey=slug,
            training_ids=train.row_ids.tolist(),
            withheld_ids=test.row_ids.tolist(),
            training_only_median_map_km3_by_scramble=volumes,
            **{k: v for k, v in receipt.items() if k != "states"},
        ),
    )


def run(args):
    design, seed, root = args
    out = root / f"{design}-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if seed:
        known(design, seed, out)
    else:
        campaign(design, out)
    return design


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--previous", action="store_true", help="Use the archived v5 model on these same inputs")
    selected = p.add_mutually_exclusive_group()
    selected.add_argument("--surveys-only", action="store_true")
    selected.add_argument("--known-only", action="store_true")
    a = p.parse_args()
    jobs = [(s, 0, a.output) for s in SURVEYS] + [
        (s, n, a.output) for s in KNOWN for n in ((90,) if s.startswith("survey_rotation") else (90, 91))
    ]
    if a.surveys_only:
        jobs = [j for j in jobs if not j[1]]
    if a.known_only:
        jobs = [j for j in jobs if j[1]]
    with ProcessPoolExecutor(a.jobs, initializer=select_engine, initargs=(a.previous,)) as pool:
        for s in pool.map(run, jobs):
            print(s, flush=True)
