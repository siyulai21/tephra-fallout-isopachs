"""GitHub-readable directional evidence and conditional extrapolation estimates."""
import json
from pathlib import Path

import numpy as np


def case_page(output,r,inventory):
    output = Path(output)
    volume = r["volume"]
    check = volume["numerical"]
    joint = volume["coherent_volume_km3"]
    mass = np.array(volume["median_map_parts_by_sector_km3"])
    audit = json.loads((output/"complete-contour-audit.json").read_text())
    bands = [k for k,v in audit["policies"].items() if v.get("selected_band")]
    display = json.loads((output/"map-display.json").read_text())
    failed = [k for k,v in r["members"].items() if not v["diagnostics"]["passed"]]
    n = len(r["data_rows"])
    rows = [f"# {r['dataset']} directional fallout", "",
        f"**{volume['median_map_km3']:,.1f} km³ of bulk fallout** is the whole-Earth integral of the median thickness map. The principal fit uses {n} accepted thickness observations and bounds and allows the fan direction to vary in unsampled sectors. Fixing fan directions with the same data and structure weights gives **{volume['matched_control_map_km3']:,.1f} km³**.", "",
        f"Under uncertain azimuth, the conditional 95% volume interval is **{joint['lower95']:,.1f}–{joint['upper95']:,.1f} km³**. Measured interpolation triangles contribute {mass[0].sum()/mass.sum():.1%} of the median-map volume. The rest relies on continuation.", "",
        "![Principal uncertain-fan-direction median isopachs](isopachs.png)", "",
        "Solid marks require a triangle whose measured sites bracket the thickness and whose posterior crossing probability reaches 90%. Dashes show the remaining median contour. White circles are thickness observations. An empty sector can contain fallout.", "",
        ("The deposit panel includes the accepted sites and the display search for the 0.1 cm contour. " +
         ("The lowest contour reaches the physical map edge in the extent check. " if not display["extent_audit"]["lowest_contour_within_chart"] else "")
         + "[Whole-Earth thickness map](world.png) · [Display extent and checks](map-display.json)"), "",
        "## Reach from the volcano", "",
        "![Fixed and uncertain fan directions near the source](median-isopachs.png)", "",
        "Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.", "",
        "| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |",
        "|---|---:|---:|---:|---:|"]
    by = {(x["method"],x["sector"]):x["first_crossing_great_circle_km"] for x in r["source_connected_reach"] if x["level_cm"]==10}
    area = volume["area_by_sector_km2"]
    j = list(volume["level_cm"]).index(10.)
    for i,sector in enumerate(volume["sectors"]):
        rows.append(f"| {sector} | {by['control',sector]:,.1f} | {by['revised',sector]:,.1f} | {area['control'][j][i]:,.0f} | {area['revised'][j][i]:,.0f} |")
    rows += ["","![Probability that a deposit reaches ten centimetres](exceedance.png)","",
        "The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.","",
        "## Withheld observations","",
        "Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. Coverage counts exact withheld thicknesses inside the uncertain-direction 95% predictive interval. A dash marks an unavailable score or coverage.", "",
        "| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing | Exact-site 95% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,v in r["validation"].items():
        a,p = v["all_estimated"],v["passed_only"]
        values = [{"geographic":"Geographic regions","quadrants":"Source quadrants"}.get(k,k)]
        for item in (a,p):
            values += ([str(item["folds"]),f"{item['control_score']:.3f}",f"{item['revised_score']:.3f}"] if item else ["0","—","—"])
        coverage = v.get("observation_interval", {})
        values.append(f"{coverage['covered_exact']}/{coverage['scored_exact']} ({coverage['coverage95_exact']:.1%})"
                      if coverage.get("scored_exact") else "—")
        rows.append("| "+" | ".join(values)+" |")
    rows += ["","[Predictions for every estimable held-out site](heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.", ""]
    if r["archive"] and r["input"].get("interpreted_upper_limits"):
        rows += ["## Inspected archives", "",
                 "The principal fit includes the interpreted upper limits below. Each core enters once. Other sites provide an occurrence check. The interval describes the modeled latent thickness at each site.", "",
                 "| Site | Interpreted upper limit (cm) | Median thickness (cm) | Conditional 95% interval (cm) |",
                 "|---|---:|---:|---:|"]
        for site in inventory["sites"]:
            local = r["archive"]["scenarios"]["quantitative"]["sites"][site["id"]]
            low, med, high = local["latent_cm_p025_p50_p975"]
            scale = site.get("principal_effective_upper_cm")
            role = f"{scale:g}" if scale is not None else "Occurrence only"
            rows.append(f"| {site['name']} | {role} | {med:.3g} | {low:.3g}–{high:.3g} |")
        rows += [""]
    elif r["archive"] and not inventory["scenarios"]:
        rows += ["## Independent archive predictions", "",
                 "These locations have reported ash occurrences or tentative correlations. The fitted thickness data supply the predictions. The intervals describe latent bulk thickness at each location.", "",
                 "| Archive | Identification | Median thickness (cm) | Conditional 95% interval (cm) |",
                 "|---|---|---:|---:|"]
        estimates = r["archive"]["scenarios"]["quantitative"]["sites"]
        for site in inventory["sites"]:
            low, median, high = estimates[site["id"]]["latent_cm_p025_p50_p975"]
            rows.append(f"| {site['name']} | {site['confidence']} | {median:.3g} | {low:.3g}–{high:.3g} |")
        rows.append("")
    elif r["archive"]:
        rows += ["## Inspected archives", "",
          "The thickness-only fit uses the quantitative table. Each visibility sensitivity asks what the deposit would look like if specified thicknesses at selected archives would have been recognizable as beds. The local preservation and detection scales are assumptions. The same coherent deposit supplies all site predictions.", "",
          "| Visibility assumption | Sihailongwan (cm) | Quanyang (cm) | Lake Kushu (cm) | Ulleung Basin (cm) | Effective draws |",
          "|---|---:|---:|---:|---:|---:|"]
        ids = [s["id"] for s in inventory["sites"]]
        named = ["longgang","quanyang","old-084","old-085"]
        if set(named).issubset(ids):
            for name,item in r["archive"]["scenarios"].items():
                values = [item["sites"][k]["latent_cm_p025_p50_p975"][1] for k in named]
                rows.append(f"| {name.replace('_',' ')} | "+" | ".join(f"{v:.3g}" for v in values)+f" | {item['effective_draws']:,.0f} |")
            site = r["archive"]["scenarios"]["quantitative"]["sites"]["longgang"]["latent_cm_p025_p50_p975"]
            exact = r["archive"]["analytic_phase_median_cm"]["longgang"]
            rows += ["",f"At Sihailongwan the thickness-only revised median is {site[1]:.2f} cm, with a conditional 95% latent range of {site[0]:.3g}–{site[2]:.3g} cm. Analytically averaging over every possible azimuth gives a median of {exact:.2f} cm. Sun and colleagues counted glass shards in centimetre-thick core slices. Neither the slice thickness nor the core depth measures the original fallout bed. [Primary core study](https://doi.org/10.1016/j.gr.2015.01.013)", ""]
        else:
            rows = rows[:-3]
            rows += ["Site-level medians, intervals, visibility likelihoods and independent-packet sensitivity are in the numerical record.",""]
        rows += ["The visibility scenarios use the recorded assumptions below.", ""]
        for s in inventory["scenarios"]:
            scale = ", ".join(f"{k} {v:g} cm" for k,v in s["effective_limit_cm"].items())
            rows.append(f"- **{s['id']}** uses {scale}. {s['assumption']}")
        if "archive_conditioned" in volume:
            q=volume["archive_conditioned"]
            v=q["coherent_volume_km3"]
            rows += ["",f"The mapped **{r['archive']['mapped_scenario']}** sensitivity gives {q['median_map_km3']:,.1f} km³ from its median map and a conditional complete-deposit interval of {v['lower95']:,.1f}–{v['upper95']:,.1f} km³."]
        rows += [""]
    unstable = [f"{a['level_cm']:g} cm ({a['symmetric_difference_fraction']:.1%})" for a in check["split_areas"]
                if a["effective_union_nodes"]>=64 and a["symmetric_difference_fraction"]>=.1]
    rows += ["## Checks and reproducibility","",
        f"The shared contour audit found eligible bands under {', '.join(bands) if bands else 'none of the screened support policies'}. Audit status is {audit['status'].replace('_',' ')}. [All contour decisions and deletion failures](complete-contour-audit.json)","",
        f"The spatial integration check {'passed' if check['spatial_passed'] else 'flagged instability'} at {check['history'][-1]['points_per_scramble']:,} points per scramble. The map used up to {r['draws_per_chain']} posterior draws per chain and structure. "
        +("Split posterior packets disagreed about the area at "+", ".join(unstable)+"." if unstable else
          ("Resolved contour areas passed the split-packet check." if check["split_map_passed"] else "The split-packet volume check flagged instability."))
        +(" The joint-volume quantiles passed the split check." if check["joint_quantile_split_passed"] else " The joint-volume quantiles changed by over 10% on splitting the draws."),"",
        "Full-fit sampling warnings occurred for "+", ".join(failed)+"." if failed else "All six full fits passed their sampling diagnostics.","",
        "[Numerical report, all sector volumes and sensitivities](report.json) · [Complete fields and per-field volumes](fields.npz) · [Principal map grids](median-grids.npz) · [Near-source comparison grids](map-grid.npz) · [Fragment decisions](fragment-decisions.csv) · [Bracketed coordinates](bracketed-fragments.csv) · [File hashes](artifacts.json)",""]
    (output/"README.md").write_text("\n".join(rows))
