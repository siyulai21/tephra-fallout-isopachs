"""Compare the previously unread Figure 7 sites with fields fitted before either visual review."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .pinatubo import GEOD, INPUT, ROOT
from .refresh_figures import recorded_observations, regional_models
from src.Laea import make_laea_transform
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.visualization.deposit_field import FAMILIES, open_field
from src.empirical_isopach.visualization.principal import volume_rows


ARCHIVE = ROOT / "archive/2026-11-22-pinatubo-intake/pre-visual-review"


def previous_fields(output):
    home = ARCHIVE / "results"
    training = ARCHIVE / "input.csv"
    previous = pd.read_csv(training)
    revised = pd.read_csv(INPUT)
    new = revised.loc[~revised.EvidenceID.isin(previous.EvidenceID)].copy()
    if not new.EvidenceID.str.startswith("P7-").all() or new.empty:
        raise ValueError("Expected previously unread Figure 7 sites")
    data, meta = recorded_observations({"path": training}, home)
    regional = json.loads((home / "report.json").read_text())
    if regional["input"]["input_sha256"] != meta["input_sha256"]:
        raise ValueError("Archived regional fit and training observations differ")
    project, _, _ = make_laea_transform(*meta["vent_lonlat"])
    xy = np.column_stack(project.transform(new.Lon.to_numpy(), new.Lat.to_numpy()))
    lookup = new[["EvidenceID", "Lon", "Lat", "Kind"]].copy()
    lookup["reported_cm"] = new.Thickness.fillna(new.Limit_cm)
    nearest, distance = [], []
    for lon, lat in zip(new.Lon, new.Lat, strict=True):
        _, _, metres = GEOD.inv(np.full(len(previous), lon), np.full(len(previous), lat),
                                previous.Lon.to_numpy(), previous.Lat.to_numpy())
        index = np.argmin(metres)
        nearest.append(previous.EvidenceID.iloc[index])
        distance.append(metres[index] / 1000)
    lookup["nearest_earlier_id"] = nearest
    lookup["nearest_earlier_km"] = distance
    models, _, _ = regional_models(data, regional)
    for name, model in models.items():
        lookup[name + "_median_cm"] = np.exp(predictive_distribution(model, xy).quantile(.5))
    for family in FAMILIES:
        field, _, _ = open_field(home / family, family, data.canonical(), meta)
        lookup[family + "_median_cm"] = 10 ** field.central(xy)
    output.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(output, index=False)
    return lookup


def volume_changes(home, output):
    before = pd.read_csv(ARCHIVE / "volumes.csv").set_index("method")
    if (home / "spherical-mixture/extent/report.json").is_file():
        # The historical comparison isolates a source-table change with both model versions held fixed.
        archived = ROOT / "archive/2026-11-22-global-extent/principal-volumes.csv"
        volumes = ("median_map_volume_km3", "conditional_volume_lower95_km3", "conditional_volume_upper95_km3")
        release = pd.read_csv(archived, converters={name: float for name in volumes})
        after = release.loc[release.slug == "pinatubo-1991-layer-c"].set_index("method")
    else:
        after = pd.DataFrame(volume_rows(home)).set_index("method")
    old_hash = sha256((ARCHIVE / "input.csv").read_bytes()).hexdigest()
    new_hash = sha256(INPUT.read_bytes()).hexdigest()
    if set(before.index) != set(after.index) or not before.input_sha256.eq(old_hash).all() or not after.input_sha256.eq(new_hash).all():
        raise ValueError("Expected five matched methods and their two audited inputs")
    out = []
    for method, a in after.iterrows():
        b = before.loc[method]
        if a.domain != b.domain:
            raise ValueError(f"Integration domain differs for {method}")
        row = {"method": method, "domain": a.domain, "before_km3": b.median_map_volume_km3,
                    "after_km3": a.median_map_volume_km3, "change_km3": a.median_map_volume_km3 - b.median_map_volume_km3,
                    "change_percent": 100 * (a.median_map_volume_km3 / b.median_map_volume_km3 - 1),
                    "before_lower95_km3": b.conditional_volume_lower95_km3, "before_upper95_km3": b.conditional_volume_upper95_km3,
                    "after_lower95_km3": a.conditional_volume_lower95_km3, "after_upper95_km3": a.conditional_volume_upper95_km3,
                    "before_input_sha256": old_hash, "after_input_sha256": new_hash}
        for check in ("map_integration_passed", "complete_draw_checks_passed", "heldout_checks_passed"):
            row["before_" + check] = b[check]
            row["after_" + check] = a[check]
        out.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume-output", type=Path, help="Also compare the two releases' median maps and conditional volume intervals")
    args = parser.parse_args()
    previous_fields(args.output)
    if args.volume_output:
        volume_changes(ROOT / "eruptions/pinatubo-1991-layer-c/results", args.volume_output)
