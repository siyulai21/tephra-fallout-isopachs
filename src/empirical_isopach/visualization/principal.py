"""Reader-facing volumes from the median and quartile thickness maps on a common input."""
import csv
from pathlib import Path

from .deposit import FAMILIES, digest, read_json
from .deposit_field import display_name
from .summary import metrics, principal_report

REGIONAL = {"stacked":"GP ensemble", "spline":"Cubic spline"}


def volume_row(key, domain, volume, band, map_ok, fit_ok, scores, input_hash):
    lo,mid,hi = band["volumes_km3"]
    if not (lo <= volume <= hi and abs(mid/volume-1) < .002):
        raise ValueError(f"The thickness-band integral disagrees with the {key} median map")
    score,coverage,validation_ok = scores
    return dict(method=key, domain=domain, median_map_volume_km3=volume,
                thickness_q25_volume_km3=lo, thickness_q75_volume_km3=hi,
                map_integration_passed=bool(map_ok), thickness_band_integration_passed=band["passed"],
                posterior_checks_passed=fit_ok, heldout_log_score=score, heldout_coverage95=coverage,
                heldout_checks_passed=bool(validation_ok), input_sha256=input_hash)


def volume_rows(home, reports=None, *, regional_home=None, input_sha256=None):
    home = Path(home)
    region = Path(regional_home or home)
    regional = reports[0] if reports else read_json(region/"report.json") if (region/"report.json").exists() else None
    rows = []
    if regional is not None:
        key_hash = regional["input"]["input_sha256"]
        bands = read_json(region/"thickness-bands.json")
        if bands["regional_report_sha256"]!=digest(region/"report.json") or bands["input_sha256"]!=key_hash:
            raise ValueError("Regional thickness bands belong to another fit")
        for key,result in regional["methods"].items():
            vol = result["regional_volume"]
            hist = [v["median_surface_km3"] for v in vol["refinement"]]
            passed = len(hist)>=3 and all(abs(a/b-1)<vol["tolerance"] for a,b in zip(hist[-3:-1],hist[-2:]))
            m = regional["validation"]["methods"][key]["metrics"]
            rows.append(volume_row(key,"observation hull",vol["median_surface_km3"],bands["methods"][key],passed,None,
                                   (m["equal_region_log_score"],m["coverage95_observations"],True),key_hash))
    for key in FAMILIES:
        folder = home/key
        if not (folder/"deposit.json").is_file():
            continue
        receipt = read_json(folder/"deposit.json")
        if receipt["family"] != key:
            raise ValueError(f"Wrong deposit family at {folder}")
        validation_reports = ("multiscale/report.json", "footprints/report.json", "extent/report.json")
        if key == "wrapped-angular-covariance":
            validation_reports = ("report.json",)
        path = next((folder / k for k in validation_reports if k in receipt["sources_sha256"]),
                    folder / "extent/report.json")
        if path.is_file():
            valid = read_json(path)["validation"]
            v = valid.get("geographic",valid)
            scores = v["negative_log_score"],v["coverage95_exact"],v["all_folds_passed"]
        else:
            # A new parent fit may be inspected before geographic continuation is installed.
            rr = reports or (regional,read_json(home/"directional/azimuth/report.json"),read_json(home/"spherical-mixture/report.json"))
            names = list(REGIONAL)+list(FAMILIES)
            scores = metrics(*rr,principal_report(home),home=home)[names.index(key)][1:]
        rows.append(volume_row(key,"whole Earth",receipt["median_map_volume_km3"],receipt["thickness_band"],
                               receipt["integration_passed"],receipt["posterior_checks_passed"],scores,receipt["input_sha256"]))
    if not rows or len({r["input_sha256"] for r in rows} | ({input_sha256} if input_sha256 else set()))!=1:
        raise ValueError(f"Missing fits or conflicting input checksums in {home}")
    return rows


def write_rows(path, rows):
    with Path(path).open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return f"{value:,.0f}" if value>=1000 else f"{value:,.3g}"


def case_section(home, prefix="", *, reports=None, regional_subdir="", name=None):
    region = Path(home)/regional_subdir
    rows = volume_rows(home,reports,regional_home=region)
    regional = any(row["domain"]=="observation hull" for row in rows)
    names = dict(REGIONAL)
    if regional:
        from ..reporting import report_label
        reg = reports[0] if reports else read_json(region/"report.json")
        names.update({key:report_label(reg,key) for key in reg["methods"] if key not in REGIONAL})
    has_minima = bool(regional and reg["input"].get("n_minima_fitted_as_measurements"))
    for row in rows:
        key = row["method"]
        if key in FAMILIES:
            receipt = read_json(Path(home)/key/"deposit.json")
            has_minima |= bool(receipt.get("observation_policy", {}).get("reported_minima_shown_as_points"))
            names[key] = display_name(key,receipt["sources_sha256"])
            name = name or receipt["eruption"]
    if name is None:
        name = (reports[0] if reports else read_json(region/"report.json"))["dataset"]
    lines = ["## Volumes and geographic prediction", "",
             "Each estimate integrates the median thickness on its map. The conditional range integrates the pointwise 25th and 75th percentiles of thickness over the same area. "
             "The 50% probability refers to thickness at each location. A single probability for total volume does not follow from these endpoints.", "",
             "| Method | Integration area | Median map (km³) | Q25–Q75 thickness maps (km³) | Held-out score ↓ | 95% predictive coverage |",
             "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        k = row["method"]
        label = f"[{names[k]}]({prefix}{k}/README.md)" if k in FAMILIES else names[k]
        ok = row["thickness_band_integration_passed"] and row["posterior_checks_passed"] is not False
        s,c = row["heldout_log_score"],row["heldout_coverage95"]
        lines.append(f"| {label} | {row['domain']} | **{number(row['median_map_volume_km3'])}**{'' if row['map_integration_passed'] else ' †'} "
                     f"| {number(row['thickness_q25_volume_km3'])}–{number(row['thickness_q75_volume_km3'])}{'' if ok else ' †'} "
                     f"| {'—' if s is None else f'{s:.3f}'}{'' if row['heldout_checks_passed'] else ' †'} "
                     f"| {'—' if c is None else f'{c:.1%}'} |")
    coverage = ("Coverage counts held-out measurements and reported minima fitted as points."
                if has_minima
                else "Coverage counts exact withheld measurements.")
    lines += ["","Lower scores indicate better geographic prediction. "+coverage+" A dagger marks a fit, integration or validation warning.", "",
              "## Full deposits and near-vent detail", "",
              "Each full map follows the thickest power-of-two contour enclosing at least 95% of the fitted locations, with a 1 cm minimum. "
              "The title gives the actual count, including when that minimum prevents 95% coverage. Global volumes include the thinner tail. "
              "The innermost line is the thickest doubling level enclosing an accepted location. The near-vent window expands from the proximal sites until eligible contours are visible. The curve sums all patches at each displayed thickness.", ""]
    for method in FAMILIES:
        if method in names:
            lines += [f"![{name} · {names[method]}. Full deposit, near-vent detail and thickness versus square root of area.]({prefix}{method}/deposit.png)", ""]
    if regional:
        regional_prefix = prefix + (regional_subdir.rstrip("/")+"/" if regional_subdir else "")
        lines += ["## Inside the observation hull", "",
                  f"![{name} · GP ensemble and cubic spline on a common scale and the full observation hull]({regional_prefix}isopachs.png)", "",
                  "The GP and spline integrate over the dashed hull. Gaps inside it can still be poorly constrained."]
        if regional_prefix:
            lines[-1] += f" [Regional fits and numerical checks]({regional_prefix}README.md)"
    return "\n".join(lines).rstrip()+"\n"
