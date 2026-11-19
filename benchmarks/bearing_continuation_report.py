"""Summarize the matched comparisons without dropping failed or empty folds."""
import argparse
import csv
import json
from pathlib import Path

from src.empirical_isopach.models.directional.bearing_calibration import PACKET


def publish(output,packet=PACKET):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    info=json.loads(Path(packet).read_text())
    table=[]
    for score in info["matched_scores"]:
        case=next(c for c in info["cases"] if c["slug"]==score["slug"])
        folds=[f for f in case["folds"] if f["kind"]==score["kind"]]
        table.append({"eruption":case["slug"],"design":score["kind"],"scored_rows":score["scored_rows"],
                      "measured_fan_negative_log_score":score["new_nll"],"six_structure_negative_log_score":score["control_nll"],
                      "change":score["new_nll"]-score["control_nll"],"warning_folds":" ".join(map(str,score["flagged_folds"])),
                      "unscored_folds":" ".join(str(f["fold"])+"/"+f["status"] for f in folds if f["status"]!="estimated"),
                      "calibration_eruptions":" ".join(score["external_calibration"]),"input_sha256":case["input_sha256"]})
    with (output/"matched-holdouts.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator="\n");writer.writeheader();writer.writerows(table)
    lines=["# Directional extrapolation in unsampled bearings","",
           "Both candidates predict the same buffered geographic and whole-quadrant holdouts from six eruptions. The measured-fan candidate uses three observation-derived fan widths and four cross-bearing range ratios. The comparator uses three widths with and without a possible fan in the largest distal gap. Both receive weights trained on the other eruptions using the same fold and eruption balance. Smaller scores indicate better prediction.","",
           "| Eruption | Geographic, measured fans / comparator | Quadrants, measured fans / comparator |",
           "|---|---:|---:|"]
    for case in info["cases"]:
        cells=[]
        for kind in ("geographic","quadrants"):
            row=next((r for r in table if r["eruption"]==case["slug"] and r["design"]==kind),None)
            cells.append("unscored" if row is None else f"{row['measured_fan_negative_log_score']:.3f} / {row['six_structure_negative_log_score']:.3f}")
        lines.append(f"| {case['slug']} | "+" | ".join(cells)+" |")
    lines+=["","The narrowed candidate scores worse on geographic holdouts in all six cases. The thickness measurements preferentially test locations where a layer was measurable. Angular coverage alone provides insufficient grounds to infer a thin or absent deposit in a blank sector. The case report therefore distinguishes field-supported portions, the measured-fan median, the possible-missing-fan sensitivity, and any plausibility check from an independently investigated archive.","",
           "The numeric comparison keeps all estimable held-out folds, including their warnings. An external fold enters weight calibration only when the compared structures jointly passed their sampling checks. Empty or unestimable sectors are retained. When checking a target, the calibration excludes its name, identical input bytes and another record at its recorded vent. The regularization has the weight of one eruption distributed across candidates.","",
           "[Matched scores, warnings and search boundaries](matched-holdouts.csv) · [Fold IDs, individual log scores, weights and posterior hashes](../../src/empirical_isopach/assets/bearing-calibration.json) · [Protocol](../../docs/methods/archive-evidence.md) · [B-Tm western-sector test](../../eruptions/b-tm/results/directional/bearing/README.md)",""]
    (output/"README.md").write_text("\n".join(lines))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();publish(args.output)
