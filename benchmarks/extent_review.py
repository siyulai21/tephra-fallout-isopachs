"""Replay the current global continuations from verified full and geographic-fold fits."""
import argparse
from pathlib import Path

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.extent_release import review_dataset
from src.empirical_isopach.models.extent_review import FAMILY_NAMES
from src.empirical_isopach.visualization.deposit import render_deposit


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",required=True,choices=[r["slug"] for r in datasets()])
    p.add_argument("--family",required=True,choices=FAMILY_NAMES)
    p.add_argument("--fit-work",type=Path,help="Directional work/ensemble or spherical posteriors from the matching fit")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Choose a fresh output directory")
    if a.family!="nested-isopachs" and not a.fit_work:
        p.error("This family needs the full and fold posterior directories")
    rec=next(r for r in datasets() if r["slug"]==a.case)
    folder=Path("eruptions")/a.case/"results"/a.family
    data,meta=audited_data(rec["path"],allow_intervals=True)
    review_dataset(folder,a.family,data,meta,rec["name"],output=a.output/"extent",work=a.fit_work,input_path=rec["path"])
    render_deposit(folder,a.family,data,meta,rec["name"],output=a.output)
