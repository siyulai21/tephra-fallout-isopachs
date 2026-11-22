"""Assemble matched complete-deposit comparisons from audited case and challenge packets."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.nested.geometry import positions, DIAMETER
from src.empirical_isopach.models.nested.study import digest

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("Directional", "Spherical", "Nested")


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def table(path, rows):
    with path.open("w", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        output.writeheader()
        output.writerows(rows)


def three(volume):
    return [volume["lower95"], volume["upper95"]]


def case_results(record):
    slug = record["slug"]
    home = ROOT/"eruptions"/slug/"results"
    nested = read(home/"nested-whole-area/report.json")
    folds = read(home/"nested-whole-area/geographic.json")["folds"]
    directional = read(home/"directional/azimuth/report.json")
    spherical = read(home/"spherical-mixture/report.json")
    key = digest(record["path"])
    if any(doc["input"]["input_sha256"] != key for doc in (nested, directional, spherical)):
        raise ValueError(f"Case inputs differ for {slug}")
    dg = directional["validation"]["geographic"]
    sv = spherical["validation"]
    for n, d, s in zip(folds, dg["folds"], sv["fits"], strict=True):
        for partition, dk, sk in (("training_rows", "train", "training_row_ids"),
                                  ("test_rows", "test", "withheld_row_ids")):
            if not n["identity"][partition] == d[dk] == s[sk]:
                raise ValueError(f"Geographic partitions differ for {slug} fold {n['fold']}")
    draws = [three(directional["volume"]["coherent_volume_km3"]),
             three(spherical["fits"]["baseline"]["volume"]), nested["conditional_volume_interval95_km3"]]
    scores = [dg["all_estimated"]["revised_score"], sv["metrics"]["equal_region_log_score"],
              nested["geographic_score"]]
    maps = [read(home/family/file)["median_map_volume_km3"] for family,file in (
        ("directional","deposit.json"), ("spherical-mixture","deposit.json"), ("nested-whole-area","median-field.json"))]
    results = [dict(eruption=record["name"], slug=slug, method=name, lower95_km3=q[0],
                    median_map_km3=volume, upper95_km3=q[-1], equal_region_log_score=score, input_sha256=key)
               for name, q, volume, score in zip(FAMILIES, draws, maps, scores, strict=True)]
    experiments = []
    for family, folder in zip(FAMILIES, ("directional", "spherical-mixture")):
        study = read(home/folder/"sensitivity-study/report.json")
        if study["input_sha256"] != key:
            raise ValueError(f"Prior sensitivity input differs for {slug}")
        paired = study["paired_scores"]
        lo, med, hi = study["sensitivity_volume_km3"]
        experiments.append(dict(eruption=record["name"], method=family, control_score=paired["reference_score"],
                                candidate_score=paired["sensitivity_score"], control_lower95_km3=study["reference_volume_km3"][0],
                                control_upper95_km3=study["reference_volume_km3"][-1], candidate_lower95_km3=lo, candidate_upper95_km3=hi,
                                numerical_checks_passed=study["passed"]))
    nested["distal_rolloff"] = read(home/"nested-whole-area/distal-taper.json")["scenarios"]
    nested["bearing_study"] = read(home/"nested-whole-area/bearing-study/report.json")
    if nested["bearing_study"]["input_sha256"] != key:
        raise ValueError(f"Nested bearing sensitivity input differs for {slug}")
    edge = home/"nested-whole-area/edge-deletion/report.json"
    if edge.is_file():
        nested["edge_deletion"] = read(edge)
        if nested["edge_deletion"]["input_sha256"] != key:
            raise ValueError(f"Edge-deletion input differs for {slug}")
    return results, experiments, nested


def sector_evidence(record, report):
    data,meta = audited_data(record["path"])
    radius,bearing = positions(data.xy,meta["vent_lonlat"][1])
    distance = DIAMETER*np.arcsin(np.minimum(1.,radius/DIAMETER))
    sector = np.floor((bearing+np.pi/8)/(np.pi/4)).astype(int)%8
    budget = report["spatial_volume_budget"]
    rows = []
    for k,name in enumerate(budget["sectors"]):
        exact = (sector==k)&~data.censored
        lo,med,hi = budget["fraction_by_sector"][k]
        rows.append(dict(slug=record["slug"],sector=name,exact_sites=int(exact.sum()),
                      bounded_sites=int(((sector==k)&data.censored).sum()),
                      nearest_exact_authalic_arc_km=float(distance[exact].min()) if exact.any() else None,
                      farthest_exact_authalic_arc_km=float(distance[exact].max()) if exact.any() else None,
                      median_volume_fraction=med,lower95_fraction=lo,upper95_fraction=hi))
    return rows


def known_row(item, label, assumption, family, quantiles, passed):
    lo, hi = quantiles[0], quantiles[-1]
    truth = item["truths"][label]["volume_km3"]
    width = np.log(hi/lo)
    outside = max(np.log(lo/truth), np.log(truth/hi), 0.)
    return dict(scenario=item["scenario"], seed=item["seed"], truth=label, method=family,
                weight=1/len(item["truths"]), truth_km3=truth, lower95_km3=lo,
                upper95_km3=hi,
                covered=lo <= truth <= hi, log_interval_width=width, log_interval_score=width+40*outside,
                numerical_checks_passed=passed, source=assumption)


def matched_challenges(known, spherical, initial=()):
    baseline = ROOT/"comparisons/spherical-mixture/bearings/known-volumes.csv"
    prior = {(r["scenario"], int(r["seed"])): r for r in csv_rows(baseline)
             if r["geometry"] == "displaced" and r["method"] == "principal" and r["truth"] == "original"}
    volumes, areas = [], []
    for item in known:
        seed, scenario = item["seed"], item["scenario"]
        for label, actual in item["truths"].items():
            def append(family, q, passed, source):
                volumes.append(known_row(item, label, source, family, q, passed))
            append("Nested", item["quantiles_km3"], item["numerics"]["passed"], "nested SMC")
            for a in actual["areas"]:
                areas.append(dict(scenario=scenario, seed=seed, truth=label, method="Nested", level_cm=a["level_cm"],
                                  false_area_km2=a["false_km2"], missed_area_km2=a["missed_km2"],
                                  iou=a["iou"], effective_union_nodes=a["effective_union_nodes"]))
            if seed in (10, 11):
                s = prior[scenario, seed]
                if not np.isclose(actual["volume_km3"], float(s["truth_km3"])):
                    raise ValueError(f"Spherical challenge truth differs for {scenario}-{seed}")
                passed = all(s[k] == "True" for k in ("sampler_passed", "spatial_passed", "draw_split_passed"))
                append("Spherical", [float(s[k]) for k in ("lower95", "upper95")], passed,
                       "published spherical-bearing challenge")
            else:
                d = item["directional"]
                last = d["integration"][-1]
                passed = all(d["member_passed"]) and max(last["scramble_change"], last["doubling_change"]) < .025
                append("Directional", d["quantiles_km3"], passed, "verified directional ensemble draws")
                if "candidate_quantiles_km3" in d:
                    append("Directional gap amount",d["candidate_quantiles_km3"],passed,"same directional draws with uncertain gap amount")
                s = spherical.get((scenario, seed))
                if s is not None:
                    key = "total_km3" if label == "specified" else "same_data_hidden_total_km3"
                    if not np.isclose(actual["volume_km3"], s["truths"][key]):
                        raise ValueError(f"Fresh spherical challenge truth differs for {scenario}-{seed}")
                    passed = s["parent"]["passed"] and s["numerical"]["spatial_passed"] and s["numerical"]["draw_split_passed"]
                    append("Spherical", [s["volume"][k] for k in ("lower95","median","upper95")], passed, "matched spherical refit")
                    if "candidate_quantiles_km3" in s:
                        append("Spherical source prior",s["candidate_quantiles_km3"],passed and s["candidate_effective_draws"]>=100,
                               "same spherical draws with source-displacement prior")
    for item in initial:
        for label,actual in item["truths"].items():
            volumes.append(known_row(item,label,"archived unrestricted slopes","Nested initial",item["quantiles_km3"],item["numerics"]["passed"]))
            for a in actual["areas"]:
                areas.append(dict(scenario=item["scenario"],seed=item["seed"],truth=label,method="Nested initial",level_cm=a["level_cm"],
                                  false_area_km2=a["false_km2"],missed_area_km2=a["missed_km2"],iou=a["iou"],effective_union_nodes=a["effective_union_nodes"]))
    for r in csv_rows(ROOT/"comparisons/directional/synthetic-areas.csv"):
        if r["method"] != "revised_median" or r["region"] != "whole_Earth":
            continue
        scenario, seed = r["design"].rsplit("-", 1)
        areas.append(dict(scenario=scenario, seed=int(seed), truth="hidden" if r["truth"] == "hidden_lobe" else "specified",
                          method="Directional", level_cm=float(r["cm"]), false_area_km2=float(r["false_positive_km2"]),
                          missed_area_km2=float(r["false_negative_km2"]), iou=float(r["intersection_over_union"]) if r["intersection_over_union"] else None,
                          effective_union_nodes=float(r["union_effective_nodes"])))
    for r in csv_rows(ROOT/"comparisons/spherical-mixture/bearings/known-areas.csv"):
        if (r["geometry"], r["model"], r["truth"], r["region"]) != ("displaced", "principal", "original", "Earth") or int(r["seed"]) not in (10,11):
            continue
        union = float(r["union_km2"])
        areas.append(dict(scenario=r["scenario"], seed=int(r["seed"]), truth="specified", method="Spherical",
                          level_cm=float(r["level_cm"]), false_area_km2=float(r["false_area_km2"]),
                          missed_area_km2=float(r["missed_area_km2"]), iou=float(r["intersection_km2"])/union if union else None,
                          effective_union_nodes=float(r["effective_union_nodes"])))
    return volumes, areas


def comparison_plot(rows, path, nested_family="Nested"):
    names = list(dict.fromkeys(r["eruption"] for r in rows))
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for i, (family, color) in enumerate(zip(("Directional", "Spherical", nested_family), ("#427687", "#bd6c31", "#694b91"), strict=True)):
        group = [r for r in rows if r["method"] == family]
        y = np.arange(len(names)) + .23*(1-i)
        lo, med, hi = np.array([[r[k] for r in group] for k in ("lower95_km3", "median_map_km3", "upper95_km3")])
        ax.hlines(y, lo, hi, color=color, lw=1.5)
        ax.scatter(med, y, color=color, marker="D", s=20, label=family, zorder=3)
    ax.set(yticks=np.arange(len(names)), yticklabels=names, xscale="log", xlabel="Whole-deposit bulk volume (km³, log scale)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.24)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.14), ncol=3, frameon=False)
    ax.set_title("Median-thickness map integrals and conditional 95% volume intervals", loc="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def corpus(path, source=None, spherical=None, initial=None, outer=None, recovery=None):
    if source is None:
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    data = {"nested":[read(p) for p in sorted(source.glob("*.json"))],
            "spherical":[read(p) for p in sorted(spherical.glob("*/spherical.json"))],
            "initial":[read(p) for p in sorted(initial.glob("*.json"))],
            "outer":{p.stem:read(p) for p in sorted(outer.glob("*.json"))},
            "recovery":[read(p) for p in sorted(recovery.glob("*.json"))]}
    path.write_bytes(gzip.compress((json.dumps(data, separators=(",",":"))+"\n").encode(), mtime=0))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--known", type=Path, help="Fresh nested challenge receipts")
    p.add_argument("--spherical-known", type=Path, help="Fresh spherical receipts for seeds 90 and 91")
    p.add_argument("--initial-known",type=Path,help="Archived slope-prior challenge receipts")
    p.add_argument("--outer",type=Path,help="Matched withheld distal fifth receipts")
    p.add_argument("--recovery",type=Path,help="Frozen synthetic fallout and local thickening tests on real survey layouts")
    a = p.parse_args()
    if len({bool(v) for v in (a.known,a.spherical_known,a.initial_known,a.outer,a.recovery)}) != 1:
        p.error("Supply all challenge directories or reuse the published raw receipts")
    a.output.mkdir(parents=True, exist_ok=True)
    evidence = a.output/"challenge-receipts.json.gz" if a.known else ROOT/"comparisons/nested-isopachs/challenge-receipts.json.gz"
    raw = corpus(evidence, a.known, a.spherical_known,a.initial_known,a.outer,a.recovery)
    rows, experiments, reports, compass = [], [], {}, []
    for record in sorted(datasets(), key=lambda r:r["slug"]):
        case, paired, nested = case_results(record)
        rows.extend(case)
        experiments.extend(paired)
        slug = record["slug"]
        nested["initial_profile"] = read(ROOT/"archive/2026-11-21-nested-initial"/slug/"report.json")
        nested["outer_fifth"] = raw["outer"][slug]
        nested["sectors"] = sector_evidence(record,nested)
        compass.extend(nested["sectors"])
        reports[slug] = nested
    volumes, areas = matched_challenges(raw["nested"], {(s["scenario"],s["seed"]):s for s in raw["spherical"]},raw["initial"])
    for filename, data in (("whole-area-cases.csv",rows), ("peer-sensitivities.csv",experiments),("sector-evidence.csv",compass),
                           ("known-volumes.csv",volumes), ("known-areas.csv",areas)):
        table(a.output/filename, data)
    comparison_plot(rows, a.output/"whole-area-volumes.png")
    from .nested_summary_report import write_report
    write_report(a.output, rows, experiments, reports, volumes, areas,raw["recovery"])
    from .nested_principal_summary import build
    build(a.output, rows)


if __name__ == "__main__":
    main()
