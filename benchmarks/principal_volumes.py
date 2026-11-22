"""Build the matched map-integral table from registered cases and the separate Pinatubo snapshot."""
import argparse
from pathlib import Path

from .nested_reach import case_home
from .nested_reach_challenge import records
from src.empirical_isopach.visualization.principal import volume_rows, write_rows


def catalogue_rows():
    rows = []
    for record in sorted(records(), key=lambda r:r["slug"]):
        slug = record["slug"]
        home = case_home(slug)
        preview = slug=="pinatubo-1991"
        for row in volume_rows(home,regional_home=home/"regional" if preview else home):
            rows.append(dict(eruption=record["name"],slug=slug,case_status="intake study" if preview else "catalogue",**row))
    return rows


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_rows(args.output,catalogue_rows())
