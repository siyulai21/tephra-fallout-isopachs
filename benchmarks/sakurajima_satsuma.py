"""Rebuild the Satsuma fall table from native site records and published text."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/sources/sakurajima-satsuma"
AUDIT = ROOT / "data/audits/Sakurajima-Satsuma"
DATA = ROOT / "data/eruptions/Sakurajima_Satsuma.csv"
GEOD = Geod(ellps="WGS84")
VENT = (130 + 39 / 60 + 24 / 3600, 31 + 35 / 60 + 33 / 3600)
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
KINDS = ("measurement", "lower_limit", "upper_limit")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
BOOKS = {
    "nishihara-2022-tables.xlsx": "3e16d82b8d2ee49782574b0ee6252d6d5ad9f817291c967df9f17db0ca8150e7",
    "gsj-737.xlsx": "3365307898961dd5baa259b3969e8cf72fed3ae50c405f6398a375a996e90109",
}
# Row-specific decisions after reading the companion methods and excavation notes.
DRAWN = set("""1A-01 1A-03 1A-04 1A-06 1A-08 1A-12 1A-18 1A-29
1B-15 1B-19 1B-20 1B-21 1B-22 1B-23""".split())
MIXED = {"1B-11", "1B-18", "1B-25", "1B-26", "1B-27"}
WRITTEN = set("""1A-09 1A-11 1A-13 1A-16 1A-17 1A-20 1A-21 1A-23 1A-25
1A-26 1A-28 1A-30 1B-03 1B-04 1B-08 1B-10 1B-24 1B-28 1B-30 1B-31
1B-32 1B-33 1B-34 1B-36 1B-37 1C-05 1C-06 1C-07 1C-08 1D-18 1D-26
1D-32 1D-37 1D-43 1D-48 1D-50 1F-02""".split())


def sheet(path, index):
    """Read original cell addresses and cached numeric values without formatting them."""
    if sha256(path.read_bytes()).hexdigest() != BOOKS[path.name]:
        raise ValueError(f"Source workbook hash differs: {path}")
    with ZipFile(path) as book:
        strings = []
        for item in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("s:si", NS):
            text = item.findall("s:t", NS) or item.findall("s:r/s:t", NS)
            strings.append("".join(t.text or "" for t in text))
        page = ET.fromstring(book.read(f"xl/worksheets/sheet{index}.xml"))
        for row in page.findall(".//s:sheetData/s:row", NS):
            cells = {}
            for cell in row.findall("s:c", NS):
                value = cell.findtext("s:v", None, NS)
                if value is not None:
                    column = re.sub(r"\d+", "", cell.attrib["r"])
                    cells[column] = strings[int(value)] if cell.get("t") == "s" else value
            if cells:
                yield int(row.attrib["r"]), cells


def coordinate(site, axis):
    degrees, minutes, seconds = (float(site[f"{axis}_{part}"]) for part in ("degrees", "minutes", "seconds"))
    return degrees + minutes / 60 + seconds / 3600


def row(identifier, site, setting, reference, lat, lon, raw, kind="", cm="", reason="",
        reported_lower="", reported_upper="", location_source="", original_report=""):
    az, _, dist = GEOD.inv(*VENT, float(lon), float(lat))
    return dict(evidence_id=identifier, site=site, setting=setting, source_cell_or_text=reference,
                latitude=format(float(lat), ".12g"), longitude=format(float(lon), ".12g"),
                location_source=location_source, source_value_cm=raw, reported_lower_cm=reported_lower,
                reported_upper_cm=reported_upper, kind=kind, fitted_cm=cm, decision=reason,
                original_report=original_report, km=round(dist / 1000, 4), bearing_deg=round(az % 360, 4),
                sector=SECTORS[int(((az + 22.5) % 360) // 45)], accepted_csv_row="")


def archaeology(site, raw):
    value = unicodedata.normalize("NFKC", raw).replace("‐", "-")
    if site in DRAWN:
        return "", "", "Profile or schematic read from a drawing, sometimes with a ruler or WebPlotDigitizer."
    if site in MIXED:
        return "", "", "Proximal composite P14. The archive does not isolate airborne thickness from possible P14B current deposits within their reported reach."
    if site == "1B-07":
        return "", "", "Fault, fissure fill and sand injection in this excavation. The available description does not isolate a demonstrably undisturbed 50 cm fallout section."
    if site == "1C-02":
        return "", "", "The 15 cm Anjo ash has a questioned Sz-S correlation."
    if site in WRITTEN:
        if site == "1B-10":
            return "lower_limit", value, "Only early airborne unit Sa-1pfa identified at this site. This thickness is a floor on the combined fall."
        if value.startswith((">", "<")):
            kind = "lower_limit" if value.startswith(">") else "upper_limit"
            return kind, value[1:], "One-sided thickness explicitly reported in the archive."
        if "-" in value:
            lower, upper = value.split("-")
            if not (0 < float(lower) < float(upper)):
                raise ValueError(value)
            return "lower_limit", lower, "Reported range across the site. Fit its lower endpoint as a floor and retain both endpoints for review."
        if not value.isdecimal() or float(value) <= 0:
            raise ValueError(f"Unexpected written value {site}: {value}")
        reason = "Positive field thickness from written excavation or specialist stratigraphic descriptions in GSJ 737."
        if site == "1B-24":
            reason = "Report 110, volume 1, p.11, layer VI. Approximately 70 cm of primary pumice-bearing deposit persists across the excavation."
        elif site == "1A-16":
            reason = "Written mean thickness of the named stratigraphic layer in the excavation's Table 2."
        return "measurement", value, reason
    reasons = {"": "No P14 value at this excavation. Horizon completeness and detection limit are unspecified.",
               "0": "No measured positive bed. No quantitative detection or recovery limit is given.",
               "(0)": "Questioned or absent P14 identification. No reported physical detection limit.",
               "obs": "Reported occurrence without thickness or diagnostic local bound.",
               "scat": "Scattered clasts in soil. Clast size and thickness of host sediment do not measure fallout thickness.",
               "tr": "Dispersed grains without a visible layer. No measured local conversion from dispersed grains to bulk fallout."}
    if value.endswith("?"):
        return "", "", "Questioned occurrence without a securely identified bed thickness."
    return "", "", reasons[value]


def ledger():
    records = []
    for n, cells in sheet(SRC / "nishihara-2022-tables.xlsx", 2):
        if n < 7 or "D" not in cells:
            continue
        raw = cells.get("T", "")
        site = cells.get("C") or cells["B"]
        kind, cm = "", ""
        reason = "No P14 reading at this outcrop. Unexposed horizons and thin deposits cannot be distinguished."
        if raw == "scat":
            reason = "Scattered material without a continuous bed or calibrated detection limit."
        elif site == "Loc.9-3":
            reason = "169 cm at the Loc.9 section where the authors document a PDC. A separate airborne-only subtotal at this exact sublocality could not be verified."
        elif raw:
            if not raw.isdecimal() or int(raw) <= 0:
                raise ValueError(raw)
            kind, cm = "measurement", raw
            reason = "Pumice-fall thickness measured by the field team in its own Table S2."
        records.append(row(f"S2-T{n}", site, "field outcrop", f"Table S2 T{n}", cells["D"], cells["E"],
                           raw, kind, cm, reason, location_source=f"Table S2 D{n}:E{n}"))
    if len(records) != 65:
        raise ValueError("Expected 65 field sections")
    for n, cells in sheet(SRC / "gsj-737.xlsx", 1):
        if n == 1:
            continue
        site, raw = cells["A"], cells.get("W", "")
        kind, cm, reason = archaeology(site, raw)
        val = unicodedata.normalize("NFKC", raw).replace("‐", "-")
        low, high = ("", "")
        if re.fullmatch(r"\d+-\d+", val):
            low, high = val.split("-")
        elif val.startswith(">"):
            low = val[1:]
        elif val.startswith("<"):
            high = val[1:]
        records.append(row(f"GSJ-{site}", cells["B"], "archaeological site", f"GSJ 737 {site} W{n}",
                           cells["D"], cells["E"], raw, kind, cm, reason, low, high,
                           f"GSJ 737 D{n}:E{n}, site location", cells.get("AB") or cells.get("AD", "")))
    if len(records) != 65 + 132:
        raise ValueError("Expected 132 archaeological sites")
    transcript = SRC / "printed-sites.csv"
    if sha256(transcript.read_bytes()).hexdigest() != "e66749fc82b5923d84bdb28f36d36c3c3dd161451772ab8cd47926ee0faca4df":
        raise ValueError("Printed coordinate and bed transcript has changed")
    with transcript.open(newline="") as stream:
        for site in csv.DictReader(stream):
            value = site["bed_value"]
            cm = format(float(value) * {"mm": .1, "cm": 1.}[site["bed_unit"]], ".12g") if value else ""
            records.append(row(site["evidence_id"], site["site"], site["setting"], site["source"],
                               coordinate(site, "latitude"), coordinate(site, "longitude"), cm, "measurement" if cm else "", cm,
                               site["interpretation"], location_source=site["location_source"], original_report=site["url"]))
    return records


def csv_bytes(columns, rows):
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def archive_probes(rows):
    """These independent locations supply predictions only. No visibility scales are assigned."""
    probes = {
        "GSJ-1D-01": ("Tawaraishi No. 1", "cryptotephra_only", "secure"),
        "GSJ-1D-49": ("Kuwanoki", "cryptotephra_only", "secure"),
        "S18-NK457": ("U1429C NK457", "occurrence_only", "secure"),
        "S21-HGY1026": ("Huguangyan HGY1026", "cryptotephra_only", "secure"),
        "A24-SG14-1554": ("Suigetsu SG14-1554", "attribution_uncertain", "tentative"),
    }
    sites = []
    for r in rows:
        if r["evidence_id"] in probes:
            name, reading, confidence = probes[r["evidence_id"]]
            sites.append(dict(id=r["evidence_id"], name=name, reading=reading, confidence=confidence,
                              latitude=float(r["latitude"]), longitude=float(r["longitude"]), archive=r["setting"],
                              reference=r["source_cell_or_text"], inspection=r["decision"], reported_fallout_limit_cm=None))
    if len(sites) != len(probes):
        raise ValueError("Missing archive record")
    return dict(schema="tephra-archive-evidence-v1", target="Primary airborne Satsuma fallout", sites=sites, scenarios=[])


def largest_gap(angles):
    angles = sorted(angles)
    if len(angles) < 2:
        return 360.
    next_angles = angles[1:] + [angles[0] + 360]
    return round(max(b - a for a, b in zip(angles, next_angles)), 4)


def source_positions(measured):
    offsets = [("JMA reference", *VENT)]
    for i, sector in enumerate(SECTORS):
        lon, lat, _ = GEOD.fwd(*VENT, i * 45, 3000.)
        offsets.append((f"3 km {sector}", lon, lat))
    result = []
    for label, lon, lat in offsets:
        angles = [GEOD.inv(lon, lat, float(r["longitude"]), float(r["latitude"]))[0] % 360 for r in measured]
        counts = Counter(SECTORS[int(((angle + 22.5) % 360) // 45)] for angle in angles)
        # Format the derived offset locations after computing every bearing at full precision.
        result.append(dict(reference=label, longitude=f"{lon:.8f}", latitude=f"{lat:.8f}",
                           **{sector: counts[sector] for sector in SECTORS},
                           max_gap_degrees=largest_gap(angles)))
    return result


def coverage_figure():
    import matplotlib.pyplot as plt

    rows = [r for r in ledger() if r["kind"]]
    fig, ax = plt.subplots(figsize=(9.3, 4.9), layout="constrained")
    for i in range(0, 8, 2):
        ax.axhspan(i - .5, i + .5, color="#f0f4f4", zorder=0)
    for kind, marker, color, label in (("measurement", "o", "#126a71", "Measured thickness"),
                                        ("lower_limit", "^", "#a34a1f", "Lower bound"),
                                        ("upper_limit", "v", "#6748a2", "Upper bound")):
        points = [r for r in rows if r["kind"] == kind]
        x = [r["km"] for r in points]
        y = []
        for r in points:
            sector = SECTORS.index(r["sector"])
            offset = (r["bearing_deg"] - sector * 45 + 180) % 360 - 180
            y.append(sector + offset / 50)
        ax.scatter(x, y, marker=marker, s=43, color=color, edgecolors="white", linewidths=.6,
                   zorder=3, label=f"{label} ({len(points)})")
    ax.set_xscale("log")
    distances = [10, 20, 30, 50, 75, 100, 150]
    ax.set_xticks(distances, labels=[str(v) for v in distances])
    ax.set(xlim=(8, 181), ylim=(7.55, -.55), yticks=range(8), yticklabels=SECTORS,
           xlabel="Distance from the Kitadake reference (km, logarithmic scale)", ylabel="Bearing")
    ax.set_title("Satsuma fallout thickness by bearing and distance", loc="left", weight="bold")
    ax.grid(axis="x", color="#becbc9", linewidth=.65)
    ax.spines[["right", "top"]].set_visible(False)
    fig.legend(loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(AUDIT / "coverage.png", dpi=180)
    plt.close(fig)


def products():
    rows = ledger()
    fitted = [r for r in rows if r["kind"] in KINDS]
    coordinates = {(r["latitude"], r["longitude"]) for r in fitted}
    if len(fitted) != len(coordinates) or len(rows) != len({r["evidence_id"] for r in rows}):
        raise ValueError("Review repeated coordinates or identifiers.")
    observed = []
    for i, r in enumerate(fitted, 2):
        r["accepted_csv_row"] = i
        exact = r["kind"] == "measurement"
        observed.append(dict(EvidenceID=r["evidence_id"], Site=r["site"], Lon=r["longitude"], Lat=r["latitude"],
                             Thickness=r["fitted_cm"] if exact else "", **{"Vent Lon":VENT[0], "Vent Lat":VENT[1]},
                             Kind=r["kind"], Limit_cm="" if exact else r["fitted_cm"]))
    accepted = csv_bytes(observed[0], observed)
    coverage = []
    measured = [r for r in fitted if r["kind"] == "measurement"]
    for sector in SECTORS:
        group = [r for r in fitted if r["sector"] == sector]
        distances = [r["km"] for r in group if r["kind"] == "measurement"]
        counts = Counter(r["kind"] for r in group)
        coverage.append(dict(sector=sector, **{kind:counts[kind] for kind in KINDS},
                             nearest_measured_km=min(distances, default=None), farthest_measured_km=max(distances, default=None),
                             measured_beyond_50_km=sum(d >= 50 for d in distances), measured_beyond_100_km=sum(d >= 100 for d in distances)))
    gaps = {}
    for distance in (0, 20, 50, 100, 150):
        gaps[distance] = largest_gap(r["bearing_deg"] for r in measured if r["km"] >= distance)
    case = dict(file=DATA.relative_to(ROOT).as_posix(), sha256=sha256(accepted).hexdigest(),
                definition="Primary airborne Satsuma (Sz-S/P14) deposit across all eruptive groups and bearings, bulk cm as preserved.",
                n_measurements=len(measured), n_upper_limits=sum(r["kind"]=="upper_limit" for r in fitted),
                n_lower_limits=sum(r["kind"]=="lower_limit" for r in fitted))
    source_urls = ("https://doi.org/10.3389/feart.2022.988373.s004", "https://www.gsj.jp/publications/pub/openfile/openfile0737.html")
    provenance = dict(sources=[dict(file=(SRC/name).relative_to(ROOT).as_posix(), sha256=checksum, url=url)
                              for (name, checksum),url in zip(BOOKS.items(), source_urls)],
                      cases={"principal":case}, vent=dict(lon=VENT[0], lat=VENT[1], meaning="JMA Kitadake summit reference on Sakurajima; actual Pleistocene vent unresolved",
                      source="https://www.data.jma.go.jp/vois/data/fukuoka/506_Sakurajima/506_index.html"),
                      maximum_empty_measured_bearing_gap_deg_by_minimum_km=gaps)
    origins = source_positions(measured)
    return {AUDIT/"source-position-check.csv":csv_bytes(origins[0], origins), DATA:accepted, AUDIT/"selection.csv":csv_bytes(rows[0], rows), AUDIT/"coverage.csv":csv_bytes(coverage[0], coverage),
            AUDIT/"archive-probes.json":(json.dumps(archive_probes(rows), indent=2, ensure_ascii=False)+"\n").encode(),
            AUDIT/"provenance.json":(json.dumps(provenance, indent=2, ensure_ascii=False)+"\n").encode()}


def verify_gsj_archive(path):
    """Cross-check the bundled cells and hand-reviewed acquisition classes against the original archive."""
    files = json.loads((SRC / "source-files.json").read_bytes())["originals"]
    checksums = {r["object"]: r["sha256"] for r in files}
    raw_url = "https://www.gsj.jp/data/openfile/no0737/gsj_openfile_report_737.zip"
    if sha256(path.read_bytes()).hexdigest() != checksums[raw_url]:
        raise ValueError("GSJ source archive fingerprint differs")
    with ZipFile(path) as archive:
        names = archive.namelist()
        book, = [name for name in names if name.endswith(".xlsx") and not name.startswith("__MACOSX/")]
        notes, = [name for name in names if name.endswith(".docx") and not name.startswith("__MACOSX/")]
        if archive.read(book) != (SRC / "gsj-737.xlsx").read_bytes():
            raise ValueError("GSJ workbook differs from the original member")
        from io import BytesIO
        raw_notes = archive.read(notes)
        if sha256(raw_notes).hexdigest() != checksums["ZIP member ending in .docx"]:
            raise ValueError("GSJ field notes fingerprint differs")
        with ZipFile(BytesIO(raw_notes)) as doc:
            root = ET.fromstring(doc.read("word/document.xml"))
    paragraphs = ["".join(n.text or "" for n in p.findall(".//{*}t")) for p in root.findall(".//{*}p")]
    parts = re.split(r"(?m)^・(1[A-Z]-[0-9]+)[　 ]", "\n".join(paragraphs))
    blocks = {k:v.split("(文献)")[0] for k, v in zip(parts[1::2], parts[2::2])}
    if len(blocks) != 132:
        raise ValueError("Incomplete GSJ site-note extraction")
    drawings = re.compile("アプリ|定規|読み取|模式柱状図|図から", re.I)
    for site in DRAWN:
        if not drawings.search(blocks[site]):
            raise ValueError(f"Expected documentation of section drawing at {site}")
    for site in WRITTEN:
        if drawings.search(blocks[site]):
            raise ValueError(f"Graphic reading needs review at {site}")
    if "Sa-1pfa" not in blocks["1B-10"] or "地割れ" not in blocks["1B-07"]:
        raise ValueError("Partial bed or fissure interpretation differs")


def verify_marine_paper(path):
    files = json.loads((SRC / "source-files.json").read_bytes())["originals"]
    source = next(r for r in files if r["source"] == "https://doi.org/10.1016/j.quaint.2011.03.046")
    if sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Expected the pinned Waikato accepted manuscript")
    text = subprocess.check_output(["pdftotext", "-raw", str(path), "-"], text=True)
    text = re.sub(r"\s+", " ", text)
    for reading in ("38.33", "56.63", "9.120-9.128 m", "8-mm thick", "lenticular", "36.35", "24.43",
                    "T-1-1) is a fine-grained vitric ash bed, 10 cm thick"):
        if reading not in text:
            raise ValueError(f"Could not verify printed marine or island reading: {reading}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate the audited tables. The default verifies them.")
    parser.add_argument("--figure", action="store_true", help="Render the accepted observations by distance and bearing.")
    parser.add_argument("--gsj-zip", type=Path, help="Also verify the original GSJ open-file archive and site notes.")
    parser.add_argument("--marine-paper", type=Path, help="Also verify the pinned Moriwaki accepted manuscript using pdftotext.")
    args = parser.parse_args()
    if args.gsj_zip:
        verify_gsj_archive(args.gsj_zip)
    if args.marine_paper:
        verify_marine_paper(args.marine_paper)
    for path, data in products().items():
        if args.write:
            path.write_bytes(data)
        elif path.read_bytes() != data:
            raise ValueError(f"Published table disagrees with source replay: {path}")
    if args.figure:
        coverage_figure()
    print("Verified the Satsuma field, excavation and distal records.")


if __name__ == "__main__":
    main()
