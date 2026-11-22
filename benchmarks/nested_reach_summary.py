"""Build the geographic reach comparison from published case fields and validation receipts."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np

from .nested_reach import case_home
from .nested_reach_challenge import records
from src.empirical_isopach.models.nested.study import digest, write_json

COMPARISON = Path("comparisons/nested-reach")


def read(path):
    return json.loads(Path(path).read_text())


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(output, cases, known, quadrants, regional):
    output.mkdir(parents=True, exist_ok=True)
    receipts = output / "validation.json.gz"
    if known is not None:
        records_by_design = {"known":[read(p) for p in sorted(known.glob("*.json"))],
                             "quadrants":[read(p) for p in sorted(quadrants.glob("*.json"))]}
        if len(records_by_design["known"]) != 18 or len(records_by_design["quadrants"]) != 4*len(records()):
            raise ValueError("Expected all known-surface and compass holdouts, including empty sectors")
        receipts.write_bytes(gzip.compress(json.dumps(records_by_design, separators=(",",":")).encode(), mtime=0))
    else:
        records_by_design = json.loads(gzip.decompress((COMPARISON / receipts.name).read_bytes()))
        if receipts.resolve() != (COMPARISON / receipts.name).resolve():
            receipts.write_bytes((COMPARISON / receipts.name).read_bytes())
    quadrants = list(records_by_design["quadrants"])
    represented = {r["slug"] for r in quadrants}
    for rec in records():
        slug = rec["slug"]
        if slug in represented:
            continue
        folder = case_home(slug) / "nested-isopachs/compass-holdouts"
        for i in range(4):
            row = read(folder / f"quadrant-{i}.json")
            if (row["slug"], row["quadrant"], row["input_sha256"]) != (slug, i, digest(rec["path"])):
                raise ValueError(f"Mismatched published compass holdout for {slug}, quadrant {i}")
            quadrants.append(row)
    quadrant_rows = []
    for item in quadrants:
        row = {k:item[k] for k in ("slug","quadrant","status")}
        q = [float(-np.mean(item[k])) for k in ("previous_log_score","corridor_log_score")] if row["status"] == "estimated" else [None,None]
        row.update(test_rows=len(item["test_rows"]), training_rows=len(item["training_rows"]),
                   previous_score=q[0], corridor_score=q[1])
        quadrant_rows.append(row)
    table(output / "compass-holdouts.csv", quadrant_rows)
    known_rows = []
    for run in records_by_design["known"]:
        for label,truth in run["truths"].items():
            for i,(family, quantiles) in enumerate(zip(run["method_order"],run["volume_km3"],strict=True)):
                a = np.asarray(truth["area_km2"])[i]
                known_rows.append({"scenario":run["scenario"],"seed":run["seed"],"truth":label,"method":family,
                    "true_km3":truth["volume_km3"],"lower95_km3":quantiles[0],
                    "upper95_km3":quantiles[2],"covered":truth["coverage"][i],
                    "false_1cm_km2":a[0,0],"missed_1cm_km2":a[0,1],"false_west_1cm_km2":a[0,2],"missed_west_1cm_km2":a[0,3]})
    table(output / "known-surfaces.csv", known_rows)
    comparison, area_rows, links = [], [], {}
    for rec in sorted(records(), key=lambda r:r["slug"]):
        slug = rec["slug"]
        home = case_home(slug) / "nested-isopachs"
        folder = cases / slug if cases else home
        j = read(folder / "report.json")
        if digest(rec["path"]) != j["input_sha256"]:
            raise ValueError(f"Case input changed for {slug}")
        broad_map = read(home.parent/"nested-whole-area/median-field.json")
        map_receipt = read(folder/"deposit.json")
        assert map_receipt["input_sha256"] == j["input_sha256"]
        new = j["conditional_volume_interval95_km3"]
        areas = np.asarray(j["paired_areas"]["area_km2"])
        qs = [x for x in quadrant_rows if x["slug"] == slug and x["status"] == "estimated"]
        comparison.append({"slug":slug,"case":rec["name"],"catalogued":slug!="pinatubo-1991","input_sha256":j["input_sha256"],
                           "whole_area_map_km3":broad_map["median_map_volume_km3"],"lower95_km3":new[0],"median_map_km3":map_receipt["median_map_volume_km3"],"upper95_km3":new[-1],
                           "previous_geographic_score":j["reference_geographic_score"],"geographic_score":j["geographic_score"],
                           "previous_quadrant_score":float(np.mean([q["previous_score"] for q in qs])),
                           "quadrant_score":float(np.mean([q["corridor_score"] for q in qs])),
                           "estimated_quadrants":len(qs),"passed_geographic_fits":sum(f["passed"] for f in j["folds"]),
                           "full_fit_passed":j["numerics"]["fit"],"volume_populations_passed":j["numerics"]["populations"],
                           "quadrature_passed":j["numerics"]["quadrature"],"complete_band_policies":len(j["supported_band_policies"])})
        for k, threshold in enumerate(j["paired_areas"]["level_cm"]):
            for i, sector in enumerate(j["sectors"]):
                exact, bound = j["site_counts_by_sector"][i]
                area_rows.append({"slug":slug,"bearing":sector,"exact":exact,"bounded":bound,"level_cm":threshold,
                                  "previous_median_map_km2":areas[0,k,i],"corridor_median_map_km2":areas[1,k,i]})
        links[slug] = os.path.relpath(home / "README.md", COMPARISON)
    table(output / "cases.csv",comparison)
    table(output / "contour-areas.csv",area_rows)
    replay = {}
    for key, path, published in (("regional","comparison.csv","regional-replay.csv"),("known","synthetic/comparison.csv","regional-known.csv")):
        ref = Path("comparisons/nested-isopachs") / published
        if regional and digest(regional/path) != digest(ref):
            raise ValueError("Regional matched benchmark differs from the signed prior comparison")
        replay[key] = {"path":ref.as_posix(),"sha256":digest(ref),"rows":len(ref.read_text().splitlines())-1}
    write_json(output / "regional-parity.json",replay)
    reg = [r for r in comparison if r["catalogued"]]
    bands = sum(bool(r["complete_band_policies"]) for r in reg)
    avg = lambda key:np.mean([r[key] for r in reg])
    lines = ["# How far can a measured isopach be continued?", "",
             "A distant core establishes fallout at its location. Extending the same thickness across an unsampled flank can contribute much more bulk volume than the core itself constrains. "
             "The paired fits below use identical accepted observations and the same nested thickness profile. The principal corridor model allows finite angular and down-range reach around positive sites and retains a broad alternative. "
             "Bounds constrain thickness through their observation likelihood. [Geographic reach model and assumptions](../../docs/methods/nested-reach.md)", "",
             "Real surveys provide the primary comparison. On Calbuco 2015 the corridor fit improved predictions on a separate field campaign and reduced the average extra span against an independently mapped 0.1 mm isopach. The Aso-4 and Pinatubo marine transfers favored the broader fit. Synthetic deposits have a separate role in testing failure modes and numerical recovery. "
             "[Independent surveys, archive bounds and field-map checks](field-surveys/README.md)", "",
             "**B-Tm has the clearest evidence for the narrower map.** Mazama improves on buffered geographic predictions and changes little on whole-quadrant predictions. Aso-4’s western footprint shrinks substantially, while independent compass predictions favor broader continuation. "
             "The paired results are published together. Global volume remains conditional on how a fan continues across unvisited ground.", "",
             "| Eruption | Corridor median-map integral [conditional 95%], km³ | Whole-area median-map integral (km³) | Buffered score, whole-area → corridor | Compass score, whole-area → corridor |",
             "|---|---:|---:|---:|---:|"]
    for r in comparison:
        name = r["case"] + (" (separate intake)" if not r["catalogued"] else "")
        lines.append(f"| [{name}]({links[r['slug']]}) | **{r['median_map_km3']:,.1f}** [{r['lower95_km3']:,.1f}–{r['upper95_km3']:,.1f}] | {r['whole_area_map_km3']:,.1f} | "
                     f"{r['previous_geographic_score']:.3f} → {r['geographic_score']:.3f} | {r['previous_quadrant_score']:.3f} → {r['quadrant_score']:.3f} |")
    lines += ["", "Each volume integrates the pointwise median-thickness map. The 95% interval uses the two tails of the volume distribution. These are bulk deposit volumes. "
              "The two scores average withheld log predictive densities with equal weight for each nonempty region. Smaller values indicate better predictions. A quadrant with no observations supplies no predictive test. "
              "[Numerical checks per case](cases.csv) · [Every held compass sector](compass-holdouts.csv) · [Contour areas by bearing](contour-areas.csv)", "",
              f"Across the {len(reg)} catalogued eruptions the mean buffered score changes from {avg('previous_geographic_score'):.3f} to {avg('geographic_score'):.3f}. "
              f"The mean whole-quadrant score changes from {avg('previous_quadrant_score'):.3f} to {avg('quadrant_score'):.3f}. "
              f"A complete contour band passes the shared support screen in {bands} of {len(reg)} cases. The separate Pinatubo Layer C snapshot is excluded from those averages.", "",
              "## What the three disputed maps show", ""]
    for slug, sector in (("b-tm","W"),("aso-4","W"),("mazama","SW")):
        row = next(r for r in area_rows if r["slug"]==slug and r["bearing"]==sector and r["level_cm"]==1.)
        name = next(r["case"] for r in comparison if r["slug"]==slug)
        lines.append(f"- **{name}, {sector}.** The median 1 cm contour encloses {row['previous_median_map_km2']:,.0f} km² in this sector under the old fit and {row['corridor_median_map_km2']:,.0f} km² under the new fit. "
                     f"The accepted table has {row['exact']} exact and {row['bounded']} bounded readings there. [Both maps and sector volumes]({links[slug]})")
    methods = [[r for r in known_rows if r["method"] == method] for method in ("whole_area_rank","corridor")]
    covered = [sum(x["covered"] for x in group) for group in methods]
    lost = [r["missed_west_1cm_km2"] for r in methods[1] if r["scenario"]=="broad_umbrella"]
    lines += ["", "## Recovery of deposits with known thickness", "",
              f"The whole-area and corridor fits covered the known total in {covered[0]}/{len(methods[0])} and {covered[1]}/{len(methods[1])} cases, respectively. "
              "The 20 truths come from 18 surveys, including two pairs with identical observations and different hidden deposits. These are development stress tests.",
              "",f"For a genuinely broad, circular deposit surveyed only on its eastern half, the two corridor fits missed {min(lost):,.0f}–{max(lost):,.0f} km² of the true western 1 cm footprint. "
              "Both conditional volume intervals included the known total. A volume interval can cover the truth while the median contour misses a real lobe. "
              "Both models also underestimated the deliberately diffuse global tail. The inherited analogue thinning prior limits both methods there.",
              "", "The earlier regional GP and cubic-spline benchmarks were rerun on matched inputs. Both the registered-case summary and the 80-row synthetic table reproduced their previous SHA-256 hashes. "
              "[Known volume and false/missed area](known-surfaces.csv) · [Per-site scores, model diagnostics and known-surface receipts](validation.json.gz) · [Regional replay](regional-parity.json)"]
    (output / "README.md").write_text("\n".join(lines)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cases",type=Path)
    p.add_argument("--known",type=Path)
    p.add_argument("--quadrants",type=Path)
    p.add_argument("--regional",type=Path)
    a = p.parse_args()
    if bool(a.known) != bool(a.quadrants):
        p.error("Supply both fresh validation directories or replay the published receipts")
    build(a.output,a.cases,a.known,a.quadrants,a.regional)


if __name__ == "__main__":
    main()
