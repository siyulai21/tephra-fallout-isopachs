"""Replay the 1991 Pinatubo Layer C observation audit from published readings."""
import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import numpy as np
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/sources/pinatubo-1991"
AUDIT = ROOT / "data/audits/Pinatubo-1991"
INPUT = ROOT / "data/eruptions/Pinatubo_1991_layer_C.csv"
GEOD = Geod(ellps="WGS84")
COMPASS = "N NE E SE S SW W NW".split()
# Graticule centres in the original GIFs. Origin is the upper left, x east and y south.
FRAMES = {
    "K1": (45.5, 265.5, 13.5, 237.5, .5, .5),
    "K2": (46.5, 266.5, 386., 610.5, .5, .5),
    "P7": (40., 914., 23., 623.5, .75, .5),
}


def position(frame, x, y):
    left, right, top, bottom, dlon, dlat = FRAMES[frame]
    return (120 + (float(x) - left) * dlon / (right - left),
            15.5 - (float(y) - top) * dlat / (bottom - top))


VENT = position("K1", 197.5, 173.5)


def src(name):
    with (SOURCE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def record(identity, collection, lonlat, kind, value, reason):
    return dict(id=identity, source=collection, longitude=round(lonlat[0], 6),
                latitude=round(lonlat[1], 6), kind=kind, value_cm=value, decision=reason)


def kilometres(a, b):
    return GEOD.inv(a["longitude"], a["latitude"], b["longitude"], b["latitude"])[2] / 1000


def components(a, b):
    known = [float(v.lstrip("<>")) for v in (a, b) if "?" not in v and not ("?" in a+b and v.startswith("<"))]
    if "?" in a+b:
        if not known:
            return "", "", "No unqualified component thickness."
        return "lower_limit", sum(known)/10, "Minimum from the single legible, unqualified sublayer."
    if a.startswith("<") and b.startswith("<"):
        return "upper_limit", sum(known)/10, "Both sublayers have reported upper limits; sum their limits."
    if "<" in a+b:
        raise ValueError("Mixed upper and measured components require interval review.")
    kind = "lower_limit" if ">" in a+b else "measurement"
    return kind, sum(known)/10, "Add the two published sublayers once; convert mm to cm."


def minutes(value):
    degrees, minutes = value.split("°")
    return float(degrees) + float(minutes)/60


def possible_repeat(row, candidate, radius, proximity_only):
    distance = kilometres(row, candidate)
    if distance > radius:
        return False
    if proximity_only or distance <= .4:
        return True
    # Independent field dots and the sublayer survey have different symbols.
    # Flag near-equal cross-survey totals at the precision of a 1 mm reading.
    a, b = row["value_cm"], candidate["value_cm"]
    return bool(a != "" and b != "" and abs(float(a) - float(b)) <= .1000001)


def cruise_position(x, y):
    """Local graticule registration of SÜDMEER Figure 2 at 160 dpi."""
    controls = src("context/suedmeer-figure-2-controls.csv")
    pixels = np.array([[float(c["page_x"]), float(c["page_y"]), 1.] for c in controls])
    lonlat = np.array([[float(c["longitude"]), float(c["latitude"])] for c in controls])
    transform = np.linalg.lstsq(pixels, lonlat, rcond=None)[0]
    return np.array([float(x), float(y), 1.]) @ transform


def cruise_cores():
    """One new text measurement and one explicit rereading of a known station."""
    rows = []
    hess = src("hess-table-1.csv")
    for r in src("suedmeer-1999-cores.csv"):
        if r["linked"]:
            site = next(h for h in hess if r["station"] in h["stations"].split("|"))
            lonlat = minutes(site["lon_deg_min_e"]), minutes(site["lat_deg_min_n"])
            kind, value = "", ""
            note = "Repeat station. The printed 8.5–9.0 cm range agrees with Hess group 5. See "+r["linked"]+"."
        else:
            lonlat = np.round(cruise_position(r["figure2_x"], r["figure2_y"]), 2)
            kind, value = "measurement", float(r["reading_cm"])
            note = "Section 3.4 reports the bulk ash thickness. Figure 2 locates the core to chart precision. Figure 29 identifies the profile."
        rows.append(record("S99-"+r["station"], "S99", lonlat, kind, value, note))
    return rows


def marine_and_airfield_rows():
    rows = []
    for r in src("wiesner-2004-table-1.csv"):
        value = float(r["bulk_mm"]) / 10
        if r["unit_i_mm"] and float(r["unit_i_mm"])+float(r["unit_ii_mm"]) != float(r["bulk_mm"]):
            raise ValueError(r["station"])
        note = "Direct bulk ash thickness in Table 1; coordinates from a cast at the numbered station."
        if r["coordinate_cast_comment"]:
            note += " The listed cast was empty and locates the station only; the thickness is the published Table 1 reading."
        rows.append(record("W04-"+r["station"], "W04", (float(r["longitude"]), float(r["latitude"])),
                           "measurement", value, note))
    for r in src("hess-table-1.csv"):
        reading = r["thickness_mm"]
        kind, value = "", ""
        if r["linked"]:
            note = "Repeated station group; see "+r["linked"]+". "+r["qualification"]
        elif r["site"] == "14":
            kind, value = "upper_limit", 2.0
            note = "Table 1 lacks a legible number. The diversity discussion names SO-114-10 among sites with ash thinner than 2 cm. Use that stated class once."
        elif reading.startswith("(") or reading in ("", "–"):
            note = r["qualification"] or "No reliable numeric primary thickness."
        elif "–" in reading or reading.startswith(">"):
            kind, value = "lower_limit", float(reading.lstrip(">").split("–")[0])/10
            note = ("Minimum of the published replicate range; both endpoints retained in the source table."
                    if "–" in reading else "Published lower limit on the underlying primary ash bed.")
            if r["qualification"]:
                note += " "+r["qualification"]+"."
        else:
            raise ValueError(r)
        rows.append(record("H01-"+r["site"].zfill(2), "H01",
                           (minutes(r["lon_deg_min_e"]),minutes(r["lat_deg_min_n"])), kind,value,note))
    for r in src("casadevall-airfields.csv"):
        if r["station"] in ("RPLL","RPLS"):
            kind, value, note = "upper_limit", 1., "June 15–16 accumulation of 0.5–1 cm at these two named airfields. The high endpoint bounds climactic Layer C within the mixed window; facility reference position."
        else:
            kind, value, note = "", "", "Grouped damage/loading range on airport surfaces and buildings; local wet loading, accumulation and clearing are not a ground Layer C section."
        rows.append(record("AIR-"+r["station"], "AIR", (float(r["longitude"]), float(r["latitude"])), kind,value,note))
    return rows


def observations(cross_map_km=2.0, *, proximity_only=False):
    rows = []
    for r in src("paladio-table-2.csv"):
        rows.append(record("P2-"+r["section"].replace(" ", "-"), "P2", position("P7", r["x"], r["y"]),
                           "measurement", float(r["whole_layer_c_cm"]), "One total for the sampled section."))
    sections = rows.copy()
    sublayers = []
    correspondence = []
    for r in src("koyaguchi-figure-3.csv"):
        a = position("K1", r["c1_x"], r["c1_y"])
        b = position("K2", r["c2_x"], r["c2_y"])
        separation = GEOD.inv(*a, *b)[2]/1000
        correspondence.append(separation)
        if separation > 1:
            raise ValueError(f"Check sublayer match at {r['id']}: {separation:.2f} km")
        kind, value, note = components(r["c1_mm"], r["c2_mm"])
        row = record(r["id"], "K3", a, kind, value, note)
        sublayers.append(row)
        rows.append(row)
        near = min(sections, key=lambda s: kilometres(s, row))
        if kind and possible_repeat(row, near, cross_map_km, proximity_only):
            row["decision"] = "Possible repeat of named section "+near["id"]
            row["kind"] = ""
    for r in src("paladio-figure-7.csv"):
        mark, reading = r["mark"], r["reading_cm"]
        if mark in ("solid", "total") and reading and "?" not in reading:
            value = float(reading.lstrip("<>"))
            if mark == "total" and reading.startswith(">"):
                kind, value, reason = "", "", "A minimum for the whole section supplies no bound on Layer C alone."
            elif mark == "total":
                kind, reason = "upper_limit", "Total section may include A or B; Layer C is at most the reported total."
            elif reading.startswith("<"):
                kind, reason = "upper_limit", "Published upper limit for Layer C."
            elif reading.startswith(">"):
                kind, reason = "lower_limit", "Published minimum for Layer C."
            else:
                kind, reason = "measurement", "Legible Layer C field marker and thickness."
        else:
            kind, value, reason = "", "", r["note"] or "Square from the sublayer survey or unresolved reading."
        row = record(r["id"], "P7", position("P7", r["x"], r["y"]), kind, value, reason)
        if kind:
            # The triangle centres can be displaced within their printed symbol.
            overlaps = [s for s in sections if kilometres(row, s) <= .8]
            overlaps += [s for s in sublayers if possible_repeat(row, s, cross_map_km, proximity_only)]
            if overlaps:
                closest = min(overlaps, key=lambda s: kilometres(s, row))
                row["kind"] = ""
                row["decision"] = "Possible repeat at the mapped location of "+closest["id"]
        rows.append(row)
    for r in src("wiesner-wang-table-1.csv"):
        reading = r["reading_cm"]
        kind, value = ("upper_limit" if reading.startswith("<") else "measurement"), float(reading.lstrip("<"))
        note = "Caliper-measured intact core ash horizon, attributed to climactic fallout by the field synthesis."
        if r["core"] in ("1", "2"):
            kind, note = "upper_limit", "Unpartitioned proximal core bed includes a fine top compared with postclimactic D."
        if r["core"] == "7":
            note = "Measured <0.1 cm in scattered patches; upper bound at this core."
        if r["station"] == "SCS-C":
            kind, note = "", "Sediment-trap loading converted via density and porosity; independent check, not bed thickness."
        rows.append(record("WW-"+r["station"], "WW1", (minutes(r["lon_deg_min_e"]), minutes(r["lat_deg_min_n"])),
                           kind, value, note))
    rows.extend(marine_and_airfield_rows())
    rows.extend(cruise_cores())
    for row in rows:
        az, _, dist = GEOD.inv(*VENT, row["longitude"], row["latitude"])
        row["km"] = round(dist/1000, 3)
        row["bearing_deg"] = round(az % 360, 3)
        row["sector"] = COMPASS[int(((az+22.5) % 360)//45)]
    fitted = [r for r in rows if r["kind"]]
    if len({(r["longitude"],r["latitude"]) for r in fitted}) != len(fitted):
        raise ValueError("Duplicate fitted coordinates")
    return rows, max(correspondence)


def csv_bytes(fields, rows):
    out = StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue().encode()


def json_bytes(obj):
    return (json.dumps(obj, indent=2, allow_nan=False)+"\n").encode()


def products():
    all_rows, registration = observations()
    selected = [r for r in all_rows if r["kind"]]
    output_rows = []
    for r in selected:
        det = r["kind"] == "measurement"
        output_rows.append({"EvidenceID": r["id"], "Site": r["id"], "Lon":r["longitude"], "Lat":r["latitude"],
                            "Thickness": r["value_cm"] if det else "", "Vent Lon":round(VENT[0],6),
                            "Vent Lat":round(VENT[1],6), "Kind":r["kind"], "Limit_cm":"" if det else r["value_cm"]})
        r["accepted_csv_row"] = len(output_rows)+1
    cooked = csv_bytes(output_rows[0].keys(), output_rows)
    fields = ["id","source","longitude","latitude","kind","value_cm","decision","km","bearing_deg","sector","accepted_csv_row"]
    exact = [r for r in selected if r["kind"]=="measurement"]
    coverage = []
    for label in COMPASS:
        group = [r for r in selected if r["sector"]==label]
        measurements = [r for r in group if r["kind"]=="measurement"]
        radii = [r["km"] for r in measurements]
        coverage.append(dict(sector=label, exact=len(measurements),
            lower=sum(r["kind"]=="lower_limit" for r in group), upper=sum(r["kind"]=="upper_limit" for r in group),
            exact_at_least_10_km=sum(v>=10 for v in radii), exact_at_least_20_km=sum(v>=20 for v in radii),
            exact_at_least_50_km=sum(v>=50 for v in radii), nearest_exact_km=min(radii,default=None),
            farthest_exact_km=max(radii,default=None)))
    gaps = {}
    for low in (0,10,20,50,100):
        angles=sorted(r["bearing_deg"] for r in exact if r["km"]>=low)
        if angles:
            gaps[str(low)] = round(float(max(np.diff(angles+[angles[0]+360]))),3)
    case={"file":str(INPUT.relative_to(ROOT)), "sha256":sha256(cooked).hexdigest(),
          "definition":"June 15 climactic Layer C including C1+C2 and survey-attributed marine equivalents; accumulated bulk fall thickness.",
          "n_measurements":len(exact),"n_upper_limits":sum(r["kind"]=="upper_limit" for r in selected),
          "n_lower_limits":sum(r["kind"]=="lower_limit" for r in selected)}
    audit={"cases":{"principal":case},"vent_lonlat":list(VENT),"reference":"Koyaguchi Figure 3 C1 star",
           "frames":FRAMES,"max_matched_sublayer_map_discrepancy_km":round(registration,3),
           "maximum_empty_bearing_gap_deg_by_minimum_exact_site_distance_km":gaps,
           "cross_survey_policy":"Exclude any K/P7 or K/P2 pair within 0.4 km or within 2 km with matching numeric value to 0.1 cm. P2 triangles remove P7 records within 0.8 km. All compilation squares excluded at source. Sensitivities are intake counts, not volume uncertainty.",
           "rows_at_cross_map_separation_km":{str(t):sum(bool(r["kind"]) for r in observations(t)[0]) for t in (.5,1,2,3)},
           "rows_if_all_nearby_cross_survey_marks_merged":{str(t):sum(bool(r["kind"]) for r in observations(t,proximity_only=True)[0]) for t in (.5,1,2,3)},
           "source_transcript_sha256":{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.glob("*.csv"))}}
    return {INPUT:cooked, AUDIT/"selection.csv":csv_bytes(fields,all_rows),
            AUDIT/"coverage.csv":csv_bytes(coverage[0].keys(),coverage),AUDIT/"provenance.json":json_bytes(audit)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare",action="store_true",help="Regenerate the four reviewed audit files")
    args=p.parse_args()
    for path, content in products().items():
        if args.prepare:
            path.write_bytes(content)
        elif path.read_bytes()!=content:
            raise ValueError(f"Replay differs: {path.relative_to(ROOT)}")
    print("Verified Pinatubo source conversions, site decisions, coordinates, coverage and accepted input.")


if __name__=="__main__":
    main()
