"""Install a reviewed directional release, preserving the previous maps and case comparisons."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from .directional_scale_maps import ARCHIVE, read_grid
from .publication import result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.visualization.deposit import (
    FAMILIES,
    align_near_views,
    digest,
    near_frame,
    read_json,
    write_json,
)
from src.empirical_isopach.visualization.principal import case_section, volume_rows, write_rows
from src.empirical_isopach.visualization.summary import saved_case_overview

MAPS = ("deposit.png", "deposit.json", "median-grid.npz", "thickness-area.csv", "README.md", "exceedance.png", "exceedance.json")


def archive(source, destination):
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def case_home(home, root):
    rows = volume_rows(home)
    write_rows(home / "principal-volumes.csv", rows)
    if (home / "report.json").is_file():
        saved_case_overview(home)
    page = root / "README.md"
    prefix = home.relative_to(root).as_posix() + "/"
    body = page.read_text()
    start = body.index("## Volumes and geographic prediction")
    if (home / "report.json").is_file():
        last = body.index("poorly constrained.", body.index("## Inside the observation hull", start))
    else:
        last = body.index(f"]({prefix}{rows[-1]['method']}/deposit.png)", start)
    end = body.find("\n", last)
    tail = body[end:].lstrip("\n") if end >= 0 else ""
    page.write_text(body[:start] + case_section(home, prefix).rstrip() + "\n\n" + tail)


def install(record, release):
    slug = record["slug"]
    root = Path("eruptions") / slug
    home = result_home(record)
    target = home / "directional"
    previous = ARCHIVE / slug
    source = release / slug
    data, meta = audited_data(record["path"])
    assert read_json(source / "deposit.json")["input_sha256"] == meta["input_sha256"]
    old = not (target / "multiscale/report.json").is_file()
    if old:
        for name in MAPS:
            archive(target / name, previous / ("report.md.txt" if name == "README.md" else name))
        archive(target / "azimuth/artifacts.json", previous / "azimuth-artifacts.json")
        archive(root / "README.md", previous / "case.md.txt")
        for name in (
            "principal-volumes.csv",
            "overview.png",
            "geographic-site-validation.csv",
            "marine-comparison.png",
            "marine-stations.csv",
            "model-comparison.csv",
            "all-sites-map.json",
            "all-sites-map.png",
            "release.json",
        ):
            archive(home / name, previous / name)
        fields = {
            f: read_grid((source if f == "directional" else home / f) / "median-grid.npz")
            for f in FAMILIES
            if (home / f / "median-grid.npz").exists()
        }
        half = near_frame(list(fields.values()), data)
        for family in fields.keys() - {"directional"}:
            if read_json(home / family / "deposit.json")["near_half_width_km"] != half:
                for name in ("deposit.json", "deposit.png"):
                    archive(home / family / name, previous / family / name)
        with np.load(target / "median-grid.npz") as grid:
            if "triangles_km" in grid:
                keys = (
                    "triangles_km",
                    "local_vertex_ids",
                    "vertices_km",
                    "inside_observation_mesh",
                    "input_sha256",
                    "sources_json",
                )
                np.savez_compressed(target / "azimuth/interpolation-mesh.npz", **{k: grid[k] for k in keys})
    shutil.copytree(source / "multiscale", target / "multiscale", dirs_exist_ok=True)
    for name in MAPS:
        shutil.copyfile(source / name, target / name)
    rel = lambda p: os.path.relpath(p, target)
    study = Path("comparisons/directional/multiscale")
    page = target / "README.md"
    with page.open("a") as stream:
        stream.write(
            f"\n[Previous and revised isopachs at three scales]({rel(study / (slug+'.png'))}) · "
            f"[Matched geographic tests]({rel(study/'README.md')}) · "
            f"[Equations and assumptions]({rel('docs/methods/directional-multiscale.md')}) · "
            f"[Previous maps]({rel(ARCHIVE/'README.md')})\n"
        )
    manifest = target / "azimuth/artifacts.json"
    hashes = read_json(manifest)
    hashes["../README.md"] = digest(page)
    write_json(manifest, hashes)
    align_near_views(home, data, meta)
    case_home(home, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    for rec in datasets():
        install(rec, args.release)
        print(rec["slug"], flush=True)
    from .principal_volumes import refresh_catalogue

    for filename in (
        "comparisons/method-scores.png",
        "comparisons/nested-isopachs/volumes.png",
        "comparisons/principal-volumes.csv",
        "comparisons/directional/README.md",
    ):
        source = Path(filename)
        dest = ARCHIVE / source
        archive(source, dest.with_name("report.md.txt") if source.suffix == ".md" else dest)
    refresh_catalogue(Path("."))
