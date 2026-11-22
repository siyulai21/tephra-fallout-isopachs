"""Recreate principal case pages, maps and held-out coverage from the released corridor packets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .nested_reach import case_home
from .nested_reach_challenge import records
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.ensemble.members import splits
from src.empirical_isopach.models.nested.reach import PROTOCOL, ReachDraws
from src.empirical_isopach.models.nested.reach_fit import signature
from src.empirical_isopach.models.nested.reach_report import case_text, heldout_coverage, principal_map, set_principal_summary
from src.empirical_isopach.models.nested.study import digest, write_json


def render(job):
    record, output = job
    home = case_home(record["slug"]) / "nested-isopachs"
    report = json.loads((home/"report.json").read_text())
    data, meta = audited_data(record["path"], allow_intervals=True)
    if (report["protocol"] != PROTOCOL or report["engine_sha256"] != signature()
            or report["input_sha256"] != meta["input_sha256"] or report["full_packet_sha256"] != digest(home/"full.npz")):
        raise ValueError(f"Principal fit identity differs for {home}")
    for i, train, test in splits(data, meta["vent_lonlat"][1], "geographic"):
        fold = report["folds"][i]
        packet = home/f"fold-{i}.npz"
        receipt = json.loads(packet.with_suffix(".json").read_text())
        if (fold["training_rows"] != train.row_ids.tolist() or fold["test_rows"] != test.row_ids.tolist()
                or receipt["posterior_sha256"] != digest(packet)):
            raise ValueError(f"Posterior or buffered partition differs for {packet}")
        fold["exact_site_coverage95"] = heldout_coverage(ReachDraws.load(packet), test)
    set_principal_summary(report)
    out = output/record["slug"]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"report.json", report)
    (out/"README.md").write_text(case_text(report, whole_area_link="../nested-whole-area/README.md"))
    principal_map(data, meta, record["name"], out, ReachDraws.load(home/"full.npz"), report)
    return record["slug"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        for slug in pool.map(render, [(r, args.output) for r in records()]):
            print(slug, flush=True)
