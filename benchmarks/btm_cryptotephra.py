"""Reproduce B-Tm archive limits from count profiles and a local peat mass budget."""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/audits/B-Tm"
OLD = ROOT / "data/eruptions/B-Tm_reviewed.csv"
NEW = ROOT / "data/eruptions/B-Tm_cryptotephra.csv"

# Original, unscaled embedded rasters. Pixel coordinates start at the top left.
# A two-pixel allowance on each picked edge covers the width of the plotted line.
# It measures reading precision only. Profile counts have additional sampling error.
SIHAI = (
    (49, 618, 1005), (50, 637, 1020), (51, 708, 1048),
    (52, 841, 1136), (53, 820, 1255), (54, 601, 1003),
)
KUSHU = (
    (130, 161, 595), (161, 193, 600), (193, 224, 663),
    (224, 264, 682), (264, 278, 971), (278, 291, 673),
)
SUIGETSU_YIELD = 25000 / 0.1  # one cm of sediment, shards cm/g per cm of bulk fallout


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def counts(right, zero, full, axis_value, *, outward=False):
    delta = 2 if outward else 0
    # Treat both axis calibration endpoints and the picked bar edge as uncertain.
    return max(0, right-zero+2*delta)*axis_value/(full-zero-2*delta)


def profile_integrals():
    bins = []
    for depth, brown_edge, clear_edge in SIHAI:
        for colour, edge, zero, full, value in (
            ("brown", brown_edge, 594, 890, 800),
            ("clear", clear_edge, 997, 1287, 3000),
        ):
            exact = {(52, "brown"):669, (53, "clear"):2657}.get((depth, colour))
            concentration = 2*(exact if exact is not None else counts(edge, zero, full, value))
            high = 2*(exact if exact is not None else counts(edge, zero, full, value, outward=True))
            bins.append(dict(site="longgang", sample=f"{depth} cm", fraction=colour,
                             width_cm=1., concentration_per_g=concentration,
                             index=concentration, upper_index=high, printed_peak=exact is not None))
    for top, bottom, edge in KUSHU:
        exact = top == 264
        # Fig. 4 uses composite depth. Its tick at 140 cm is y=130; 150 cm is y=264.
        # Table 1 of Chen et al. (2019) gives the peak concentration and the width.
        width = 1. if exact else (bottom-top)/13.4
        high_width = 1. if exact else 10*(bottom-top+4)/(134-4)
        concentration = 30671 if exact else counts(edge, 562, 963, 30000)
        high = 30671 if exact else counts(edge, 562, 963, 30000, outward=True)
        bins.append(dict(site="old-084", sample=f"Fig. 4 pixels {top}-{bottom}",
                         fraction="all glass", width_cm=width, concentration_per_g=concentration,
                         index=width*concentration, upper_index=high_width*high, printed_peak=exact))
    return bins


def unrounded_limits():
    """Unrounded local equivalents at the declared reference response or full retention."""
    bins = profile_integrals()
    upper = {sid:sum(row["upper_index"] for row in bins if row["site"] == sid)/SUIGETSU_YIELD
             for sid in ("longgang", "old-084")}
    # Han et al. (2019), 63, 64, 65 cm. Xing et al. (2025), QY3 dry-density maximum.
    # Yang et al. (2021), lowest published bulk-density scenario in their magnitude calculation.
    upper["quanyang"] = 3 * .127 / .6
    return upper


def interpreted_limits():
    upper = unrounded_limits()
    return {sid:math.ceil(upper[sid]*100)/100 for sid in ("longgang", "quanyang", "old-084")}


def make_input(path, limits):
    existing = rows(OLD)
    inventory = json.loads((AUDIT / "archive-observations.json").read_text())
    stations = {s["id"]:s for s in inventory["sites"]}
    if set(limits) & {r["EvidenceID"] for r in existing}:
        raise ValueError("Archive already has a numeric observation")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=existing[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
        for sid, limit in limits.items():
            site = stations[sid]
            writer.writerow(dict(Lat=site["latitude"], Lon=site["longitude"], Kind="upper_limit",
                                 Limit_cm=f"{limit:g}", EvidenceID=sid, Site=site["name"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    limits = interpreted_limits()
    make_input(args.output / NEW.name, limits)
    make_input(args.output / "one_tenth_response.csv", {k:10*v for k,v in limits.items()})
    make_input(args.output / "recognition_analogs.csv", dict(longgang=1., quanyang=2., **{"old-084":1.}))
    bins = profile_integrals()
    with (args.output / "count-profiles.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=bins[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(bins)
    print(json.dumps({"upper_cm":limits, "sha256":digest(args.output / NEW.name)}, indent=2))


if __name__ == "__main__":
    main()
