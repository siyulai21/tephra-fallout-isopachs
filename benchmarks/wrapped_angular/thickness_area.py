"""Compare exponential regimes, power-law and Weibull extrapolation from identical contours."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classical_thinning import Power, Segments, choose_count, models
from .area_laws import SCREENS, supported_band


def packets(path, case, with_draws=True):
    packets = [dict(np.load(path / f"{case}-{s}.npz")) for s in (51071, 68389)]
    a, b = packets
    for k in ("earth_km2", "input_sha256", "loglevels", "brackets"):
        np.testing.assert_array_equal(a[k], b[k])
    source_file = path / "source-draws.npz"
    if source_file.is_file() and with_draws:
        with np.load(source_file) as source:
            for data, seed in zip(packets, (51071, 68389), strict=True):
                with np.load(path / f"draws-{case}-{seed}.npz") as outside:
                    for part in (outside, source):
                        np.testing.assert_array_equal(data["loglevels"], part["loglevels"])
                    if outside["field"].item() != source["field"].item():
                        raise ValueError("Regional draws and source draws differ")
                    for key in ("area", "inside", "total"):
                        combined = outside[key][:, None] + source[key]
                        data[key] = combined.reshape((-1,) + outside[key].shape[1:])
                    data["regional_draws"] = len(outside["total"])
    return packets


def curve_status(f):
    if isinstance(f, Power):
        if abs(f.exponent - 2.0) <= 32 * np.finfo(float).eps:
            return "both_ends_diverge"
        return "infinite_distal_mass" if f.exponent <= 2 else "infinite_proximal_mass"
    if isinstance(f, Segments):
        if f.rates[-1] < 0:
            return "increasing_distal_segment"
        return "flat_distal_segment" if f.rates[-1] == 0 else "finite"
    if f.rate == 0:
        return "unidentified_weibull_scale"
    if f.shape > 2:
        return "weibull_mode_inside_smallest_fitted_area"
    return "infinite_proximal_mass" if f.shape == 0 else "finite"


def suite(x, t):
    curves = models(x, t)
    validation = {}
    for kind in ("exp", "joined"):
        selected, errors = choose_count(x, t, shared=kind == "joined")
        validation.update({f"{kind}-{k}": v for k, v in errors.items()})
        if selected is not None:
            curves[f"{kind}-selected"] = curves[f"{kind}-{selected}"]
    return curves, validation


def volume_rows(info, x, t, inside, earth):
    # The stored arrays run from the outer to the inner contour.
    if not np.all(x > 0) or len(np.unique(x)) < 3:
        return [dict(**info, status="insufficient_positive_distinct_ranks", levels=len(x))]
    curves, validation = suite(x, t)
    rows = []
    for name, f in curves.items():
        tail = f.integral(x[0], earth, t[0])
        r = dict(
            **info,
            law=name,
            status=curve_status(f),
            levels=len(x),
            distinct_ranks=len(np.unique(x)),
            anchor_cm=t[0],
            anchor_sqrt_area_km=x[0],
            mapped_inside_km3=inside,
            tail_earth_km3=tail,
            hybrid_earth_km3=inside + tail,
            hybrid_infinite_km3=inside + f.integral(x[0], anchor_cm=t[0]),
            fitted_zero_to_earth_km3=f.integral(0, earth),
            fitted_zero_to_infinity_km3=f.integral(0),
            fitted_inner_rank_to_earth_km3=f.integral(x[-1], earth),
            rmse_log=np.sqrt(np.mean((f.predict_log(x) - np.log(t)) ** 2)),
        )
        if isinstance(f, Segments):
            r.update(
                segments=len(f.rates),
                distal_segment_ranks=f.counts[-1],
                counts=json.dumps(f.counts.tolist()),
                junctions_km=json.dumps(f.starts[1:].tolist()),
                rates_per_km=json.dumps(f.rates.tolist()),
                increasing_segments=int(np.sum(f.rates < 0)),
                rolling_error=validation.get(f"{name.split('-')[0]}-{len(f.rates)}", np.nan),
            )
        if isinstance(f, Power):
            r["exponent"] = f.exponent
            for factor in (2, 4, 8, 16):
                r[f"hybrid_stop_{factor}x_anchor_km3"] = inside + f.integral(
                    x[0], min(earth, factor * x[0]), t[0]
                )
        if name.startswith("weibull"):
            r.update(
                weibull_shape=f.shape,
                weibull_reference=f.reference,
                weibull_reference_rate=f.rate,
            )
        rows.append(r)
    return rows


def maps(home, output):
    rows = []
    checks = []
    for source in ("fields", "eruptions"):
        for path in sorted((home / source).iterdir()):
            a, b = packets(path, "heat", with_draws=False)

            def avg(key):
                return (a[key] + b[key]) / 2

            levels = a["loglevels"]
            curve = [
                (
                    "wrapped_median",
                    avg("median_area"),
                    avg("median_inside"),
                    float(avg("median_total")),
                )
            ]
            for j, label in enumerate(a.get("truth_labels", [])):
                curve.append(
                    (
                        "oracle_" + str(label),
                        avg("truth_area")[j],
                        avg("truth_inside")[j],
                        a["truth_total"][j],
                    )
                )
            for screen in SCREENS:
                indices = supported_band(levels, avg("support"), a["brackets"], screen)
                info = dict(dataset=source, network=path.name, screen=screen)
                if not len(indices):
                    rows.append(
                        dict(
                            **info,
                            window="full",
                            field="wrapped_median",
                            status="unsupported_band",
                        )
                    )
                    continue
                if screen == "hull99":
                    for scramble, packet in zip((51071, 68389), (a, b), strict=True):
                        ranks = np.sqrt(packet["median_area"][indices])
                        if min(ranks) > 0 and len(np.unique(ranks)) >= 3:
                            checks.extend(
                                volume_rows(
                                    dict(**info, scramble=scramble),
                                    ranks,
                                    np.exp(levels[indices]),
                                    packet["median_inside"][indices[0]],
                                    np.sqrt(packet["earth_km2"]),
                                )
                            )
                for field, area, inside, direct in curve:
                    for window, fraction in (
                        ("full", 1.0),
                        ("outer_half", 0.5),
                        ("outer_third", 1 / 3),
                    ):
                        if screen != "hull99" and window != "full":
                            continue
                        ix = indices[
                            levels[indices] <= levels[indices[0]] + fraction * np.ptp(levels[indices]) + 1e-10
                        ]
                        if len(ix) < 3:
                            continue
                        lookup = dict(**info, field=field, window=window, direct_km3=direct)
                        rows.extend(
                            volume_rows(
                                lookup,
                                np.sqrt(area[ix]),
                                np.exp(levels[ix]),
                                inside[ix[0]],
                                np.sqrt(a["earth_km2"]),
                            )
                        )
    df = pd.DataFrame(rows)
    pd.DataFrame(checks).to_csv(output / "classical-quadrature.csv", index=False)
    df.to_csv(output / "classical-volumes.csv", index=False)
    four = df.query("screen=='hull99' and field=='wrapped_median' and window=='full'")
    print(
        four[four.law.isin(["exp-selected", "power", "weibull"])]
        .pivot(index="network", columns="law", values="hybrid_earth_km3")
        .round(2)
        .to_string()
    )


def draw_intervals(home, output):
    rows = []
    for source in ("fields", "eruptions"):
        for path in sorted((home / source).iterdir()):
            a, b = packets(path, "heat")
            ix = supported_band(a["loglevels"], a["support"] + b["support"], a["brackets"], "hull99")
            if not len(ix):
                continue
            t = np.exp(a["loglevels"][ix])
            ar = (a["area"][:, ix] + b["area"][:, ix]) / 2
            inside = (a["inside"][:, ix[0]] + b["inside"][:, ix[0]]) / 2
            earth = np.sqrt(a["earth_km2"])
            point = np.sqrt((a["median_area"][ix] + b["median_area"][ix]) / 2)
            chosen = choose_count(point, t)[0]
            joined_choice = choose_count(point, t, shared=True)[0]
            unique, inv = np.unique(ar, axis=0, return_inverse=True)
            masses = {}
            limits = {}
            for j, area in enumerate(unique):
                if min(area) <= 0 or len(np.unique(area)) < 3:
                    continue
                fields = models(np.sqrt(area), t)
                if f"exp-{chosen}" in fields:
                    fields["exp-selected"] = fields[f"exp-{chosen}"]
                if f"joined-{joined_choice}" in fields:
                    fields["joined-selected"] = fields[f"joined-{joined_choice}"]
                mask = inv == j
                for name, f in fields.items():
                    tail = f.integral(np.sqrt(area[0]), earth, t[0])
                    masses.setdefault(name, []).extend(inside[mask] + tail)
                    limits[name] = limits.get(name, 0) + int(mask.sum()) * (
                        curve_status(f)
                        in (
                            "infinite_distal_mass",
                            "both_ends_diverge",
                            "flat_distal_segment",
                            "increasing_distal_segment",
                            "unidentified_weibull_scale",
                        )
                    )
            for name, values in masses.items():
                lo, median, hi = np.quantile(values, [0.025, 0.5, 0.975])
                rows.append(
                    dict(
                        dataset=source,
                        network=path.name,
                        law=name,
                        selected_segments={
                            "joined-selected": joined_choice,
                            "exp-selected": chosen,
                        }.get(name, np.nan),
                        n_regional=int(a.get("regional_draws", len(ar))),
                        n_requested=len(ar),
                        n_valid=len(values),
                        nonintegrable_or_limit=limits[name],
                        q025=lo,
                        median=median,
                        q975=hi,
                    )
                )
    pd.DataFrame(rows).to_csv(output / "classical-draw-intervals.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=Path("comparisons/wrapped-angular-covariance/thickness-area")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--draws",
        action="store_true",
        help="Include conditional volume intervals over whole geographic fields",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    maps(args.inputs, args.output)
    if args.draws:
        draw_intervals(args.inputs, args.output)
