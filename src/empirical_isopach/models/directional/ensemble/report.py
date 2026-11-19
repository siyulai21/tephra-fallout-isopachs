"""Publish the ensemble median surface, its integral and coherent-field uncertainty."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from ....core.catalog import dataset_record
from ...mass_mixture.study import audited_data
from ..geometry import COMPASS, polar
from ..integration import PARTS
from ..study import digest, write_json
from .fields import Ensemble, PER_CHAIN, load_members
from .integrals import coherent, median_integral
from .maps import render_maps
from .members import MEMBERS, NAMES, directory
from .validation import support_audit, validate
from .weights import PACKET, calibrate


def _sector_counts(data, geom):
    radius, angle = polar(data.xy, geom.axis)
    index = np.floor((angle+np.pi/8)/(np.pi/4)).astype(int)%8
    return [{key:data.row_ids[(radius>0)&(index==j)&mask].tolist() for key,mask in
             (("exact_row_ids",~data.censored),("bound_row_ids",data.censored))} for j in range(8)]


def modeled_mass(data, latitude, work, weighting, full=True):
    low, high = np.quantile(data.log_value[~data.censored], [.05,.95])
    levels = np.unique(np.exp(np.linspace(low,high,7)))
    rounds = []
    for count in (PER_CHAIN, 2*PER_CHAIN, 4*PER_CHAIN):
        model = Ensemble(load_members(data, latitude, work, n=count),weighting["weights"])
        joint, mass = coherent(model,data,latitude,work,full)
        centre = median_integral(model,joint["field_mean_volume_km3"],levels)
        numerical = centre["numerical"]
        rounds.append({"draws_per_chain":count,"volume_km3":centre["volume_km3"],**numerical})
        if numerical["draw_split_passed"] and numerical["resolved_level_splits_passed"]:
            break
    centre["refinement"] = rounds
    return model, joint, mass, centre


def save_draws(output, model, mass):
    packet, volumes = {}, {}
    for member, a in zip(model.members,mass,strict=True):
        name = member.geometry.policy.name
        for k,v in member.draws.items():
            packet[name+"__"+k] = v
        packet[name+"__draw_index"] = member.index
        volumes[name] = a.reshape(member.draws["source"].shape[0],-1,5,8)
    np.savez_compressed(output/"map-fields.npz", **packet)
    np.savez_compressed(output/"coherent-mass.npz", **volumes)


def formatted(q):
    return f"{q['median']:,.1f} [{q['lower95']:,.1f}–{q['upper95']:,.1f}]"


def narrative(output, r, *, family_page=False):
    """Write the fixed-direction report, including the earlier family-root layout."""
    prefix = "ensemble/" if family_page else ""
    c, j, w = r["median_surface"], r["coherent_fields"], r["weighting"]
    n, m, g = c["numerical"], np.array(c["parts_sectors_km3"]), r["map"]
    full = [s.replace("_"," ") for s,v in j["members"].items() if not v["passed"]]
    lines = [f"# {r['dataset']} directional ensemble", "",
        f"**{c['volume_km3']:,.1f} km³ of bulk fallout** is the whole-Earth integral of the median thickness map below. "
        f"Across complete simulated deposits the median volume is **{j['volume_km3']['median']:,.1f} km³**, with a conditional 95% interval of "
        f"**{j['volume_km3']['lower95']:,.1f}–{j['volume_km3']['upper95']:,.1f} km³**.", "",
        f"The measured triangles contain **{m[0].sum():,.1f} km³** ({m[0].sum()/m.sum():.1%}) of the median map. "
        "The remaining volume depends on how thickness continues through gaps, toward the vent and beyond the sites. "
        "Each mapped contour encloses the median thickness level. Lobes and holes can remain separate.", "",
        f"![Measured-bracketed median fragments and expanded median isopachs]({prefix}isopachs.png)", "",
        f"The star marks the supplied vent. Its median thickness is {g['vent_median_cm']:,.1f} cm. "
        f"The largest median thickness on the two display grids is {g['max_on_display_grids_cm']:,.1f} cm at a source distance of "
        f"{g['max_source_distance_chord_km']:.1f} km. The evidence panel contains {g['displayed_triangle_levels']} measured-bracketed triangle and level pairs. "
        "The right panel includes conditional continuation.", "",
        f"![Local median thickness and pointwise 95% uncertainty]({prefix}local.png)", "",
        f"![Median thickness over the Earth]({prefix}world.png)", "",
        "## Direction and observational support", "",
        "| Direction | Exact rows | Bounds | Median-map volume (km³) | Complete fields, median [95%] (km³) |",
        "|---|---:|---:|---:|---:|"]
    for i, s in enumerate(COMPASS):
        counts = r["sector_observations"][i]
        lines.append(f"| {s} | {len(counts['exact_row_ids'])} | {len(counts['bound_row_ids'])} | {m[:,i].sum():,.1f} | {formatted(j['sectors'][i])} |")
    lines += ["", "| Domain | Median-map volume (km³) | Complete fields, median [95%] (km³) |","|---|---:|---:|"]
    for i, label in enumerate(("Measured triangles","Transition at triangle edges","Proximal background","Background between sampled radii","Distal background")):
        lines.append(f"| {label} | {m[i].sum():,.1f} | {formatted(j['parts'][i])} |")
    lines += ["","The median-map entries add to its global volume before rounding. Each simulated deposit uses one structure everywhere. "
              "Pointwise thickness bounds describe uncertainty at a location. The volume interval comes from integrating each complete field.", "",
              "## Assumptions and weights","",
              "| Fan half-width | Additional fan in largest distal gap | Predictive weight | Complete-field volume, median [95%] (km³) | Fit and mass checks |",
              "|---:|---|---:|---:|---|"]
    for policy, weight in zip(MEMBERS,w["weights"],strict=True):
        item = j["members"][policy.name]
        status = "passed" if item["passed"] else "**review**"
        lines.append(f"| {policy.half_width_deg:g}° | {'Possible' if policy.hidden_fan else 'Absent from this structure'} | {weight:.1%} | {formatted(item['volume_km3'])} | {status} |")
    s = c["weight_sensitivity_km3"]
    lines += ["",f"Equal weights give a median-map volume of **{s['uniform']:,.1f} km³**. Restricting to the three structures without an extra fan gives "
              f"**{s['without_extra_fan']:,.1f} km³**. Allowing an extra fan in all three gives **{s['with_extra_fan']:,.1f} km³**. "
              "These possibilities share the same observations. A fan missed entirely by sampling remains weakly identified.", "",
              "Independent calibration cases with usable folds were "+(", ".join(w["calibration_cases"]) or "unavailable, so all structures received equal weight")+". "
              "Weights describe held-out thickness prediction. They depend on the small eruption catalogue.", "",
              "## Validation and numerical checks","",
              "| Held-out design | Rows scored | Ensemble score ↓ | Equal weights ↓ | Single reference ↓ | Exact-site 95% coverage |",
              "|---|---:|---:|---:|---:|---:|"]
    for k,v in r["validation"].items():
        scores = ["—" if v[key] is None else f"{v[key]:.3f}"
                  for key in ("equal_region_log_score","uniform_score","reference_score")]
        coverage = v["metrics"].get("coverage95_observations")
        coverage_text = "—" if coverage is None else f"{coverage:.1%}"
        lines.append(f"| {k.capitalize()} | {v['scored_rows']}/{v['total_rows']} | " + " | ".join(scores+[coverage_text]) + " |")
    for k,v in r["validation"].items():
        bad = [str(f["fold"]) for f in v["fits"] if f["status"]=="estimated" and not f["passed"]]
        omit = [f"{f['fold']} ({f['status']})" for f in v["fits"] if f["status"]!="estimated"]
        lines += ["",f"{k.capitalize()} sampling warnings occurred in "+("folds "+", ".join(bad) if bad else "zero folds")+". "+("Unscored folds were "+", ".join(omit)+"." if omit else "All four folds produced predictions.")]
    latest = n["history"][-1]
    bands = sum(v is not None for v in r["complete_bands"].values())
    lines += ["",("Full-data structures requiring numerical review were "+", ".join(full)+". " if full else "All six full-data structures passed their numerical checks. ")+
              f"The whole-Earth spatial check {'passed' if n['spatial_passed'] else '**requires review**'} using {latest['points_per_scramble']:,} points in each of two scrambles. "
              f"The interleaved posterior-draw check {'passed' if n['draw_split_passed'] else '**requires review**'} "
              f"(largest change {latest['split_draw_central_relative_change']:.1%}). The resolved isopach-area split checks "
              f"{'passed' if n['resolved_level_splits_passed'] else '**require review**'}. "
              f"The independent same-fields mean-integral discrepancy was {latest['independent_same_draw_mean_relative_error']:.2%}.", "",
              f"The shared complete-contour audit found **{bands} qualifying policies out of {len(r['complete_bands'])}**. "
              "The conditional global estimate remains sensitive to preservation, source-region thickness, distal thinning and possible unseen fans.", "",
              f"[Numerical report, fold identities and weight sensitivity]({prefix}report.json) · [Complete contour decisions]({prefix}complete-contour-audit.json) · "
              f"[All fragment decisions]({prefix}fragment-decisions.csv) · [Mapped fragment vertices]({prefix}supported-fragments.csv) · [Held-out predictions]({prefix}heldout.csv)", "",
              f"[Reconstruct the median field]({prefix}map-fields.npz) · [Map grids and pointwise quantiles]({prefix}median-grids.npz) · "
              f"[Joint volume by chain, support class and compass sector]({prefix}coherent-mass.npz) · [Checksums]({prefix}artifacts.json)", ""]
    if family_page:
        reference = "single-structure/README.md" if (output.parent/"single-structure/README.md").is_file() else "single-structure.md"
        lines += [f"[Single-structure sensitivity study and unmeasured proximal cap]({reference})", ""]
    if not family_page:
        lines[0] = f"# {r['dataset']} · fixed-direction ensemble"
    page = output.parent/"README.md" if family_page else output/"README.md"
    page.write_text("\n".join(lines))
    return page


def render(path, work, output, name=None, *, family_page=False):
    data, meta = audited_data(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rec = dataset_record(path)
    w = calibrate(meta, rec["slug"] if rec else None)
    latitude = meta["vent_lonlat"][1]
    model, joint, mass, centre = modeled_mass(data,latitude,work,w)
    save_draws(output,model,mass)
    validation = {kind:validate(data,latitude,work,w["weights"],kind) for kind in ("geographic","quadrants")}
    audit = support_audit(data,latitude,work,model,validation["geographic"])
    write_json(output/"complete-contour-audit.json",audit)
    name = name or (rec["name"] if rec else Path(path).stem)
    maps = render_maps(output,name,data,meta,model)
    table = pd.DataFrame({"row_id":data.row_ids,"log_observation_or_limit":data.log_value,
                          "upper_bound":data.censored&~data.lower_censored,"lower_bound":data.lower_censored})
    for k,v in validation.items():
        table[k+"_fold"] = v["folds"]
        for label,values in v["predictions"].items():
            table[k+"_"+label] = values
    table.to_csv(output/"heldout.csv",index=False)
    r = {"protocol":"directional-median-ensemble-v1","dataset":name,
         "input":{k:v for k,v in meta.items() if k!="transform_inv"},
         "calibration_sha256":digest(PACKET),"implementation_sha256":{p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
         "weighting":w,"median_surface":centre,"coherent_fields":joint,"map":maps,
         "map_draws_per_chain":len(model.members[0].index),"validation":validation,"part_order":PARTS,"compass_order":COMPASS,
         "sector_observations":_sector_counts(data,model.members[0].geometry),
         "complete_bands":{k:v["selected_band"] for k,v in audit["policies"].items()},
         "designs":{p.name:json.loads((directory(work,p)/"design.json").read_text()) for p in MEMBERS},
         "policies":[asdict(p) for p in MEMBERS]}
    write_json(output/"report.json",r)
    page = narrative(output, r, family_page=family_page)
    files = ("report.json","isopachs.png","world.png","local.png","median-grids.npz","map-fields.npz",
             "coherent-mass.npz","heldout.csv","fragment-decisions.csv","supported-fragments.csv","complete-contour-audit.json")
    hashes = {f:digest(output/f) for f in files}
    hashes["../README.md" if family_page else "README.md"] = digest(page)
    write_json(output/"artifacts.json",hashes)
    if family_page:
        # Earlier runs placed a second copy of this generated report beside the arrays.
        (output/"README.md").unlink(missing_ok=True)
    return r


def reuse_reference_fits(reference_work, ensemble_work):
    """Import existing immutable fit attempts. run_members checks every design and draw digest."""
    src, dest = Path(reference_work), Path(ensemble_work)
    pairs = [(src/"fits"/n,dest/"fits"/n) for n in NAMES]
    for k in ("geographic","quadrants"):
        pairs += [(p,dest/k/p.name/"reference") for p in (src/k).glob("fold-*")]
    for a,b in pairs:
        if not (a/"design.json").is_file() or (b/"design.json").exists():
            continue
        b.mkdir(parents=True,exist_ok=True)
        for f in [a/"design.json", *a.glob("attempt-*.json"), *a.glob("attempt-*.npz"), a/"mass.npz", a/"integral.json"]:
            if f.is_file():
                shutil.copyfile(f,b/f.name)
