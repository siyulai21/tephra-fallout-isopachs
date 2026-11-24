"""Install reviewed nested releases and rebuild matched five-family displays from their source reports."""
import argparse
from pathlib import Path
import re
import shutil

import pandas as pd

from .nested_footprints import cases
from .nested_summary import comparison_plot
from .principal_volumes import catalogue_rows
from src.empirical_isopach.visualization.deposit import read_json
from src.empirical_isopach.visualization.principal import case_section, volume_rows, write_rows
from src.empirical_isopach.visualization.summary import all_case_scores, principal_report, saved_case_overview


def published_folder(case, record):
    home = Path("eruptions")/record["slug"]/"results"
    if case == record["slug"] and record.get("published_input"):
        home = home/"primary-ranges"
    return home/"nested-isopachs"


def install(case, record, source):
    target = published_folder(case, record)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source/"footprints", target/"footprints", dirs_exist_ok=True)
    for name in ("deposit.png", "deposit.json", "thickness-area.csv", "README.md"):
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
    if target.parent.name == "primary-ranges":
        return
    home = target.parent
    saved_case_overview(home)
    write_rows(home/"principal-volumes.csv", volume_rows(home))
    page = home.parent/"README.md"
    body = page.read_text()
    first = body.index("## Volumes and geographic prediction")
    end = body.index("[Regional fits and numerical checks](results/README.md)",first) + len("[Regional fits and numerical checks](results/README.md)")
    page.write_text(body[:first]+case_section(home,"results/").rstrip()+body[end:])


def write_index(base):
    lines = ["# Bayesian nested dispersal footprints", "",
             "Volumes integrate each eruption’s median-thickness map across Earth. The interval comes from coherent posterior deposits. The same geographic prior applies to all nine eruptions, including those with large unobserved sectors. [Model equations](../../docs/methods/nested-footprints.md) · [Matched geographic experiments, known deposits and maps](geography/README.md)", "",
             "| Eruption | Accepted localities | Bulk volume (km³) | Conditional 95% interval (km³) | Geographic score ↓ | Exact coverage | Geographic fits passed |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for slug,r in sorted(cases().items()):
        if r["slug"]!=slug:
            continue
        folder=published_folder(slug,r)
        j=read_json(folder/"footprints/report.json")
        d=read_json(folder/"deposit.json")
        v=j["validation"]
        lo,hi=j["conditional_interval95_km3"]
        def number(x):
            return f"{x:,.0f}" if x>=1000 else f"{x:,.3g}"
        qualified="" if j["draw_checks_passed"] else " · provisional"
        link="../../"+folder.as_posix()+"/README.md"
        lines.append(f"| [{r['name']}]({link}) | {j['accepted_rows']} | **{number(d['median_map_volume_km3'])}** | {number(lo)}–{number(hi)}{qualified} | {v['negative_log_score']:.3f} | {v['coverage95_exact']:.1%} | {v['passed_folds']}/4 |")
    lines += ["", "The current Mazama row uses the 300-locality primary-range audit. The three-family comparison below uses the preceding 271-locality revision for all three methods. Both Mazama inputs are fitted under all three nested geographic laws in the paired study. [Paired estimates and input hashes](geography/paired-results.csv) · [Matched three-family table](cases.csv)", "",
              "![Three complete-deposit families on identical inputs, with conditional intervals](volumes.png)", "",
              "[Original whole-area model](whole-area-study.md) · [Station-corridor field experiment](../nested-reach/README.md) · [Earlier maps and numerical results](../../archive/2026-11-23-nested-geography/README.md)", ""]
    (base/"README.md").write_text("\n".join(lines))


def summaries(output=Path(".")):
    output=Path(output)
    (output/"comparisons/nested-isopachs").mkdir(exist_ok=True,parents=True)
    (output/"eruptions").mkdir(exist_ok=True,parents=True)
    rows = catalogue_rows()
    write_rows(output/"comparisons/principal-volumes.csv",rows)
    base = output/"comparisons/nested-isopachs"
    table = pd.DataFrame(rows).set_index(["slug","method"])
    original = Path("archive/2026-11-23-nested-geography/cases.csv")
    family = pd.read_csv(original)
    all_reports, homes = [], {}
    listing = Path("eruptions/README.md")
    body = listing.read_text()
    data = cases()
    for slug, rec in sorted(data.items()):
        if slug!=rec["slug"]:
            continue
        home = Path("eruptions")/slug/"results"
        name = rec["name"]
        all_reports.append((name,read_json(home/"report.json"),read_json(home/"directional/azimuth/report.json"),
                            read_json(home/"spherical-mixture/report.json"),principal_report(home)))
        homes[name] = home
        row = table.loc[(slug,"nested-isopachs")]
        mask = (family.slug==slug) & (family.method=="Nested corridor")
        family.loc[mask,["method","median_map_km3","lower95_km3","upper95_km3","equal_region_log_score"]] = [
            "Nested footprints",row.median_map_volume_km3,row.conditional_volume_lower95_km3,row.conditional_volume_upper95_km3,row.heldout_log_score]
        volumes = [table.loc[(slug,k),"median_map_volume_km3"] for k in ("directional","spherical-mixture","nested-isopachs")]
        values = [f"{v:,.0f}" if v>=1000 else f"{v:,.3g}" for v in volumes]
        if rec.get("published_input"):
            latest = read_json(published_folder(slug,rec)/"deposit.json")["median_map_volume_km3"]
            current_rows = read_json(published_folder(slug,rec)/"footprints/report.json")["accepted_rows"]
            matched_rows = principal_report(home)["accepted_rows"]
            values[2]=f"{latest:,.0f} ({current_rows} sites), {values[2]} ({matched_rows})"
        pattern=r"(\| \[[^\n]*?\]\("+re.escape(slug)+r"/README\.md\) \|) [^|]* \| [^|]* \| [^|]* \|"
        body,n=re.subn(pattern,lambda m:m[1]+" "+" | ".join(values)+" |",body)
        if n!=1:
            raise ValueError(slug)
    body = body.replace("Nested corridors","Nested footprints").replace("Nested footprintss","Nested footprints")
    (output/listing).write_text(body)
    family.to_csv(base/"cases.csv",index=False)
    comparison_plot(family.to_dict("records"),base/"volumes.png",nested_family="Nested footprints")
    all_case_scores(output/"comparisons",all_reports,homes=homes)
    write_index(base)


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
