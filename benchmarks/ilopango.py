"""Replay the Ilopango Unit G audit directly from the authors' spreadsheet and printed core readings."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re

from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/sources/ilopango-tbj"
AUDIT = ROOT / "data/audits/Ilopango-TBJ"
INPUT = ROOT / "data/eruptions/Ilopango_TBJ_Unit_G.csv"
BOOK = SOURCES / "pedrazzi-supplement-1.xls"
CORES = SOURCES / "smith-2020-cores.csv"
BOOK_HASH = "0b9cf7846ed313bfb0383edae7088c97ab6df9f7390d24ddce588b79f8db6403"
VENT = (-89.053, 13.672)
GEOD = Geod(ellps="WGS84")
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
GENERALIZED = {"G1", "F1", "F2", "F3", "S1", "S3"}
FIELDS = ("EvidenceID", "Site", "Lon", "Lat", "Thickness", "Vent Lon", "Vent Lat", "Kind", "Limit_cm")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def original(value):
    return format(value, ".15g") if isinstance(value, float) else str(value).strip()


def coordinate(text, hemisphere):
    match = re.fullmatch(r"\s*([NW])\s*(\d+)\s*°\s*(\d+(?:\.\d+)?)\s*'\s*", text)
    if not match or match[1] != hemisphere or float(match[3]) >= 60:
        raise ValueError(f"Invalid degree-minute coordinate: {text!r}")
    degrees = int(match[2]) + float(match[3]) / 60
    return round(-degrees if hemisphere == "W" else degrees, 8)


def decision(site, value):
    if site in GENERALIZED:
        return "", "", "Regional label and rounded coordinates without an identifiable outcrop in the workbook."
    if site == "289":
        return "", "", "Tazumal has 20 cm in P67 and 40–50 cm in the Figure 3o caption. The section correspondence is unresolved."
    if isinstance(value, float):
        if value > 0:
            return "measurement", original(value), "Positive measured Unit G thickness in column P."
        if value == 0:
            return "", "", "Unit G has a zero entry. Section completeness and a positive visibility limit are undocumented."
        raise ValueError("Unexpected negative thickness")
    label = original(value)
    if re.fullmatch(r">\d+(?:[.,]\d+)?", label):
        return "lower_limit", label[1:].replace(",", "."), "Published lower limit on Unit G thickness."
    if not label or "?" in label:
        return "", "", "Unspecified or questioned thickness. No unqualified measured endpoint."
    raise ValueError(f"Unreviewed reading at {site}: {label}")


def setting(site, row):
    if site.startswith(("SO173-", "M66-")):
        return "marine core"
    if site in {"LA", "LY", "PP"}:
        return "lake or pond"
    if row >= 105:
        return "compiled terrestrial locality"
    return "outcrop"


def ledger():
    import xlrd  # Only the audit replay needs the original BIFF spreadsheet reader.
    if digest(BOOK) != BOOK_HASH:
        raise ValueError("The original workbook differs from the audited source.")
    sheet = xlrd.open_workbook(BOOK).sheet_by_name("Hoja1")
    rows = []
    for r in (*range(9, 81), *range(85, 101), *range(105, 115), *range(116, 125)):
        cells = sheet.row_values(r-1)
        site = original(cells[0])
        kind, thickness, reason = decision(site, cells[15])
        row = dict(evidence_id=f"SM1-P{r}", source="Pedrazzi et al. 2019, Hoja1", source_row=r,
                   site_id=site, site=original(cells[4]), setting=setting(site, r),
                   latitude_as_published=original(cells[2]), longitude_as_published=original(cells[3]),
                   latitude=coordinate(cells[2], "N"), longitude=coordinate(cells[3], "W"),
                   units_at_section=original(cells[5]), total_tbj_cm=original(cells[7]),
                   unit_g_cm=original(cells[15]), kind=kind, fitted_cm=thickness, decision=reason)
        if site == "LY":
            row["decision"] += " Two cores represented by one published lake coordinate and one limit."
        if site == "303":
            row["decision"] += " Section also lists a lahar; P records the identified Unit G bed."
        if r in range(105,115):
            row["decision"] += " Kutterolf et al. (2008) correlate the C1 marine bed with TBJ. Its Unit G assignment follows Pedrazzi et al."
        rows.append(row)
    with CORES.open(newline="") as f:
        for core in csv.DictReader(f):
            rows.append(dict(evidence_id="S20-"+core["core"],source="Smith et al. 2020, Ash Dispersal Modeling",
                source_row="",site_id=core["core"],site=core["core"],setting="distal core",
                latitude_as_published=core["latitude_n"]+" N",longitude_as_published=core["longitude_w"]+" W",
                latitude=float(core["latitude_n"]),longitude=-float(core["longitude_w"]),
                units_at_section="distal TBJ",total_tbj_cm=core["thickness_cm"],unit_g_cm="",
                kind="measurement",fitted_cm=core["thickness_cm"],
                decision="Directly reported distal bed; the authors used it in the co-ignimbrite reconstruction."))
    for row in rows:
        az, _, metres = GEOD.inv(*VENT, row["longitude"], row["latitude"])
        row["bearing_deg"] = round(az % 360, 4)
        row["distance_km"] = round(metres/1000, 4)
        row["sector"] = SECTORS[int(((az+22.5) % 360)//45)]
    if len(rows) != 109 or len({r["evidence_id"] for r in rows}) != len(rows):
        raise ValueError("Source rows must be unique and exhaustive.")
    return rows


def csv_data(columns, rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def measurement(row):
    exact = row["kind"] == "measurement"
    return {"EvidenceID":row["evidence_id"],"Site":row["site_id"],"Lon":row["longitude"],
            "Lat":row["latitude"],"Thickness":row["fitted_cm"] if exact else "",
            "Vent Lon":VENT[0],"Vent Lat":VENT[1],"Kind":row["kind"],
            "Limit_cm":"" if exact else row["fitted_cm"]}


def angular_gap(angles):
    ordered = sorted(angles)
    if not ordered:
        return 360.
    return round(max(b-a for a,b in zip(ordered, ordered[1:]+[ordered[0]+360])), 4)


def products():
    rows = ledger()
    accepted = [r for r in rows if r["kind"]]
    if len({(r["longitude"], r["latitude"]) for r in accepted}) != len(accepted):
        raise ValueError("Repeated accepted coordinates require a site decision.")
    for i, r in enumerate(accepted, 2):
        r["accepted_csv_row"] = i
    exact = [r for r in accepted if r["kind"] == "measurement"]
    coverage = []
    for sector in SECTORS:
        group = [r for r in exact if r["sector"]==sector]
        km = [r["distance_km"] for r in group]
        coverage.append(dict(sector=sector, measured=len(group),
            lower_limits=sum(r["sector"]==sector and r["kind"]=="lower_limit" for r in accepted),
            nearest_measured_km=min(km,default=None),farthest_measured_km=max(km,default=None),
            measured_at_least_20_km=sum(v>=20 for v in km),measured_at_least_100_km=sum(v>=100 for v in km)))
    input_csv = csv_data(FIELDS, [measurement(r) for r in accepted])
    case = dict(file=str(INPUT.relative_to(ROOT)),sha256=sha256(input_csv).hexdigest(),
        definition="Widespread co-ignimbrite fallout, Pedrazzi Unit G and explicitly correlated distal beds.",
        n_measurements=len(exact),n_lower_limits=len(accepted)-len(exact),n_upper_limits=0)
    provenance = {"sources":[
        {"file":str(BOOK.relative_to(ROOT)),"sha256":BOOK_HASH,"article":"https://doi.org/10.1016/j.jvolgeores.2019.03.006",
         "download":"https://ars.els-cdn.com/content/image/1-s2.0-S0377027318303706-mmc1.xls",
         "sheet":"Hoja1","thickness":"Column P, centimetres","site_rows":107},
        {"file":str(CORES.relative_to(ROOT)),"sha256":digest(CORES),"article":"https://doi.org/10.1073/pnas.2003008117",
         "location":"Methods, Ash Dispersal Modeling","sites":2}],
        "cases":{"principal":case},"vent":{"lon":VENT[0],"lat":VENT[1],"meaning":"Ilopango caldera reference",
         "source":"https://volcano.si.edu/volcano.cfm?vn=343060"},
        "maximum_empty_bearing_gap_deg_by_minimum_measured_distance_km":{
            str(minimum):angular_gap(r["bearing_deg"] for r in exact if r["distance_km"]>=minimum)
            for minimum in (0,20,50,100,200)},
        "exact_measurements_by_setting":dict(Counter(r["setting"] for r in exact))}
    ledger_keys = [k for k in rows[0] if k != "accepted_csv_row"]+["accepted_csv_row"]
    return {INPUT:input_csv, AUDIT/"selection.csv":csv_data(ledger_keys,rows),
            AUDIT/"coverage.csv":csv_data(coverage[0],coverage),
            AUDIT/"provenance.json":(json.dumps(provenance,indent=2,allow_nan=False)+"\n").encode()}


def coverage_figure():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with (AUDIT / "selection.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["kind"]]
    fig, ax = plt.subplots(figsize=(9, 4.9), layout="constrained")
    for row in rows:
        index = SECTORS.index(row["sector"])
        within_sector = (float(row["bearing_deg"]) - index * 45 + 180) % 360 - 180
        # Place near-coincident sites at their measured angular offsets.
        y = index + within_sector / 56
        lower = row["kind"] == "lower_limit"
        ax.scatter(float(row["distance_km"]), y, s=57 if lower else 36,
                   marker="^" if lower else "o", facecolor="white" if lower else "#1b6570",
                   edgecolor="#99491f" if lower else "white", linewidth=1.4 if lower else .65,
                   zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 100, 200, 350], labels=["10","20","30","50","100","200","350"])
    ax.set(xlim=(9, 395), ylim=(7.55, -.55), yticks=range(8), yticklabels=SECTORS,
           xlabel="Distance from Ilopango caldera (km, logarithmic scale)", ylabel="Bearing")
    ax.grid(axis="x", color="#c9d2d0", linewidth=.75)
    for i in range(0, 8, 2):
        ax.axhspan(i-.5, i+.5, color="#f1f5f4", zorder=0)
    ax.spines[["right","top"]].set_visible(False)
    ax.set_title("Ilopango Unit G  |  Where thickness was observed", loc="left", weight="bold")
    handles = [Line2D([], [], ls="", marker="o", color="#1b6570", label="Measured thickness (52 sites)"),
               Line2D([], [], ls="", marker="^", markerfacecolor="white", color="#99491f",
                      label="Reported lower bound (4 sites)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    fig.savefig(AUDIT / "coverage.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true",help="Write the four reproducible audit products.")
    parser.add_argument("--figure", action="store_true", help="Draw coverage from the verified site ledger.")
    args=parser.parse_args()
    for path,content in products().items():
        if args.prepare:
            path.write_bytes(content)
        elif path.read_bytes()!=content:
            raise ValueError(f"Saved file differs from workbook replay: {path.relative_to(ROOT)}")
    if args.figure:
        coverage_figure()
    print("Verified Ilopango workbook rows, printed core locations, thicknesses, limits and geographic coverage.")


if __name__ == "__main__":
    main()
