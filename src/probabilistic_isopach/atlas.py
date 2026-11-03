"""Build a consistent, offline atlas for every bundled deposit."""
import argparse
import json
from pathlib import Path

from .catalog import datasets
from .publication import write_atlas_index


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build closed-isopach maps, checked regional volumes and an offline atlas.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", help="Names/slugs from data/datasets.json; defaults to all four.")
    parser.add_argument("--map-grid", type=int, default=481)
    parser.add_argument("--volume-grid", type=int, default=48)
    parser.add_argument("--method", choices=("selected", "stacked"), default="selected")
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-extent-km", type=float, default=6000.)
    parser.add_argument("--skip-validation", action="store_true", help="Omit independent nested validation for a faster exploratory build.")
    args = parser.parse_args(argv)
    rows = datasets()
    if args.datasets:
        requested = {name.lower() for name in args.datasets}
        known = {name for row in rows for name in (row["name"].lower(), row["slug"])}
        if requested-known:
            parser.error(f"Unknown datasets: {sorted(requested-known)}")
        rows = [row for row in rows if requested & {row["name"].lower(), row["slug"]}]
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory; previous atlas builds are preserved.")
    args.output.mkdir(parents=True, exist_ok=True)
    from .cli import main as fit
    reports = []
    for row in rows:
        output = args.output/row["slug"]
        print(f"Analysing {row['name']} ({len(reports)+1}/{len(rows)})", flush=True)
        command = [str(row["path"]), "--output", str(output), "--name", row["name"],
                   "--map-grid", str(args.map_grid), "--volume-grid", str(args.volume_grid),
                   "--draws", str(args.draws), "--seed", str(args.seed), "--method", args.method,
                   "--max-extent-km", str(args.max_extent_km), "--atlas-parent"]
        if not args.skip_validation:
            command.append("--nested-cv")
        fit(command)
        reports.append((row["slug"], json.loads((output/"report.json").read_text())))
    write_atlas_index(args.output, reports)
    print(f"Atlas ready: {(args.output/'index.html').resolve()}", flush=True)
