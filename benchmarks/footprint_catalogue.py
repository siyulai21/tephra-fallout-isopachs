"""Install reviewed nested releases and rebuild matched five-family displays from their source reports."""
import argparse
from pathlib import Path
import shutil


from .nested_footprints import cases
from src.empirical_isopach.visualization.deposit import read_json
from src.empirical_isopach.visualization.principal import case_section, volume_rows, write_rows
from src.empirical_isopach.visualization.summary import saved_case_overview


def published_folder(case, record):
    if case != record["slug"]:
        raise ValueError(f"Expected the registered case {record['slug']}, received {case}")
    return Path("eruptions")/case/"results/nested-isopachs"


def install(case, record, source):
    target = published_folder(case, record)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source/"footprints", target/"footprints", dirs_exist_ok=True)
    for name in ("deposit.png", "deposit.json", "median-grid.npz", "thickness-area.csv", "README.md"):
        shutil.copyfile(source/name, target/name)
    if not (target/"footprints/archive-probes.json").is_file():
        from src.empirical_isopach.core.audit import audited_data
        from src.empirical_isopach.models.nested.footprints.posterior import FootprintField
        from src.empirical_isopach.models.nested.footprints.release import archive_checks
        data, meta = audited_data(record["path"])
        archive_checks(record["path"],target,data,meta,FootprintField.load(target/"footprints/full-selected.npz"))
    if not (target/"report.json").exists():
        shutil.copyfile(source/"footprints/report.json", target/"report.json")
    import os
    def link(path):
        return os.path.relpath(path, target)
    study = link("comparisons/nested-isopachs/geography/README.md")
    protocol = link("docs/methods/nested-footprints.md")
    old = link("archive/2026-11-23-nested-geography/README.md")
    page = target/"README.md"
    page.write_text(page.read_text().split("\n## Geographic support and extrapolation")[0].rstrip()+"\n")
    with page.open("a") as stream:
        if "footprints/archive-probes.json" not in page.read_text():
            stream.write("\n[Predicted thickness at audited archive sites](footprints/archive-probes.json)\n")
        stream.write(f"\n[Footprint model and assumptions]({protocol}) · [Matched geography and volume tests]({study}) · [Previous corridor maps and original whole-area controls]({old})\n")
    from src.empirical_isopach.models.nested.footprints.release import describe_geography, sector_checks
    j = read_json(target/"footprints/report.json")
    if not (target/"footprints/sector-volumes.csv").exists():
        from src.empirical_isopach.core.audit import audited_data
        from src.empirical_isopach.models.nested.footprints.posterior import FootprintField
        data, _ = audited_data(record["path"])
        sector_checks(target,data,FootprintField.load(target/"footprints/full-selected.npz"),j)
    describe_geography(target,j)
    home = target.parent
    from src.empirical_isopach.core.audit import audited_data
    from src.empirical_isopach.visualization.deposit import align_near_views
    align_near_views(home,*audited_data(record["path"]))
    rows = volume_rows(home)
    write_rows(home/"principal-volumes.csv", rows)
    if (home/"report.json").is_file():
        saved_case_overview(home)
    root = Path("eruptions")/record["slug"]
    page = root/"README.md"
    prefix = home.relative_to(root).as_posix()+"/"
    body = page.read_text()
    start = body.index("## Volumes and geographic prediction")
    if (home/"report.json").exists():
        last = body.index("## Inside the observation hull",start)
        last = body.index("poorly constrained.",last)
    else:
        last_family = rows[-1]["method"]
        last = body.index(f"]({prefix}{last_family}/deposit.png)",start)
    end = body.find("\n",last)
    tail = body[end:].lstrip("\n") if end>=0 else ""
    page.write_text(body[:start]+case_section(home,prefix).rstrip()+"\n\n"+tail)


def summaries(output=Path(".")):
    from .principal_volumes import refresh_catalogue
    refresh_catalogue(output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=Path)
    parser.add_argument("--case",action="append")
    parser.add_argument("--summaries",action="store_true")
    args=parser.parse_args()
    if args.release:
        for case,record in cases().items():
            if not args.case or case in args.case:
                install(case,record,args.release/case)
    if args.summaries:
        summaries()
