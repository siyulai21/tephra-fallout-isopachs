"""Reader-facing volumes from the five principal median surfaces, with conditional draws kept separate."""
import csv
from pathlib import Path

from .deposit import FAMILIES, volume_interval, read_json
from .summary import metrics


def volume_rows(home, reports=None, *, regional_home=None):
    home = Path(home)
    region = Path(regional_home or home)
    if reports is None:
        reports = (read_json(region/"report.json"),read_json(home/"directional/azimuth/report.json"),
                   read_json(home/"spherical-mixture/report.json"))
    regional, directional, spherical = reports
    nested = read_json(home/"nested-isopachs/report.json")
    values = []
    for key in ("stacked","spline"):
        result = regional["methods"][key]
        vol = result["regional_volume"]
        q = result["joint_volume"]["conditional_quantiles_km3"]
        hist = [v["median_surface_km3"] for v in vol["refinement"]]
        passed = len(hist)>=3 and all(abs(a/b-1)<vol["tolerance"] for a,b in zip(hist[-3:-1],hist[-2:]))
        values.append((key,"observation hull",vol["median_surface_km3"],[q[k] for k in ("0.025","0.975")],
                       passed,result["joint_volume"]["expected_grid_converged"]))
    for (key, report) in zip(FAMILIES,(directional,spherical,nested),strict=True):
        receipt = read_json(home/key/"deposit.json")
        if receipt["input_sha256"]!=regional["input"]["input_sha256"] or receipt["family"]!=key:
            raise ValueError("A principal deposit uses a different input or family")
        ext = home/key/"extent/report.json"
        q,passed = volume_interval(key,report,read_json(ext) if ext.exists() else None)
        values.append((key,"whole Earth",receipt["median_map_volume_km3"],q,receipt["integration_passed"],passed))
    rows = []
    for (key,domain,volume,q,map_ok,draw_ok),(name,score,coverage,validation_ok) in zip(
            values,metrics(*reports,nested,home=home),strict=True):
        rows.append(dict(method=key,domain=domain,median_map_volume_km3=volume,
                         conditional_volume_lower95_km3=q[0],conditional_volume_upper95_km3=q[1],
                         map_integration_passed=bool(map_ok),complete_draw_checks_passed=bool(draw_ok),
                         heldout_log_score=score,heldout_coverage95=coverage,heldout_checks_passed=bool(validation_ok),
                         input_sha256=regional["input"]["input_sha256"]))
    return rows


def write_rows(path, rows):
    with Path(path).open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def case_section(home, prefix="", *, reports=None, regional_subdir=""):
    region = Path(home)/regional_subdir
    rows = volume_rows(home,reports,regional_home=region)
    name = (reports[0] if reports else read_json(region/"report.json"))["dataset"]
    names = {"stacked":"GP ensemble","spline":"Cubic spline",**FAMILIES}
    lines = ["## Volumes and geographic prediction", "",
             "Each volume integrates its method’s median-thickness map. The GP and spline cover the same observation hull. "
             "The other three methods cover Earth, including fallout below 0.5 cm. Conditional 95% intervals describe variation in whole-deposit volume across posterior draws.", "",
             "| Method | Median-map volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        k = row["method"]
        label = names[k]
        if k in FAMILIES:
            label = f"[{label}]({prefix}{k}/README.md)"
        score = row["heldout_log_score"]
        cov = row["heldout_coverage95"]
        lo,hi = row["conditional_volume_lower95_km3"],row["conditional_volume_upper95_km3"]
        vol = row["median_map_volume_km3"]
        number = f"{vol:,.0f}" if vol>=1000 else f"{vol:.3g}"
        lines.append(f"| {label} | **{number}**{'' if row['map_integration_passed'] else ' †'} "
                     f"| {lo:,.3g}–{hi:,.3g}{'' if row['complete_draw_checks_passed'] else ' (provisional)'} "
                     f"| {'—' if score is None else f'{score:.3f}'}{'' if row['heldout_checks_passed'] else ' †'} "
                     f"| {'—' if cov is None else f'{cov:.1%}'} |")
    lines += ["","Lower scores indicate better predictions in withheld geographic regions. Coverage counts exact withheld measurements. "
              "A dagger marks a numerical or geographic validation warning. Provisional intervals retain their fit or integration warnings.", "",
              "## Full deposits and near-vent detail", "",
              "The outer line is 0.5 cm. Contours and observation colours double from there. All three methods use the same near-vent window for this eruption. "
              "Each curve counts the combined area of all patches at or above a thickness.", ""]
    for method,title in FAMILIES.items():
        lines += [f"![{name} · {title}. Full deposit, near-vent detail and thickness versus square root of area.]({prefix}{method}/deposit.png)", ""]
    regional_prefix = prefix + (regional_subdir.rstrip("/")+"/" if regional_subdir else "")
    lines += ["## Inside the observation hull", "",
              f"![{name} · GP ensemble and cubic spline on a common scale and the full observation hull]({regional_prefix}isopachs.png)", "",
              "The dashed boundary encloses all accepted sites. Gaps inside it can still be poorly constrained."]
    if regional_prefix:
        lines[-1] += f" [Regional fits and numerical checks]({regional_prefix}README.md)"
    return "\n".join(lines)+"\n"
