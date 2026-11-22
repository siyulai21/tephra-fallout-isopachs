"""Join Pinatubo source readings and contextual surveys into one reviewable evidence inventory."""
import csv
import json
from pathlib import Path

from pyproj import Geod, Transformer

from .pinatubo import AUDIT, SOURCE, src

GEOD = Geod(ellps="WGS84")


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def schematic():
    """Two city controls locate the schematic. Populated places independently check its scale."""
    places = {r["name"]:r for r in read(SOURCE/"context/geonames-luzon.csv")}
    forward=Transformer.from_crs(4326,32651,always_xy=True)
    inverse=Transformer.from_crs(32651,4326,always_xy=True)
    def anchor(name):
        row=places[name]
        return complex(*forward.transform(float(row["longitude"]),float(row["latitude"])))
    a,b=complex(175,-414),complex(332,-596)
    u,v=anchor("Angeles City"),anchor("San Fernando")
    result=[]
    for r in src("david-table-1.csv"):
        code=r["site"]
        p=complex(float(r["point_x"]),-float(r["point_y"]))
        projected=u+(p-a)*(v-u)/(b-a)
        lon,lat=inverse.transform(projected.real,projected.imag)
        place={"Telebastagen":"Telabastagan","Mabalacat":"Mabalacat City"}.get(code,code)
        check=places.get(place)
        result.append(dict(site=code,source_x=r["point_x"],source_y=r["point_y"],coarse_unit_cm=r["coarse_unit_cm"],
                           approximate_lon=round(lon,4),approximate_lat=round(lat,4),
                           check_place=place if check else "",check_geoname_id=check["geoname_id"] if check else "",
                           distance_to_named_settlement_km=round(abs(anchor(place)-projected)/1000,2) if check else ""))
    return result


def transcripts():
    original={}
    for r in src("paladio-table-2.csv"):
        original["P2-"+r["section"].replace(" ","-")]=(r["section"],r["whole_layer_c_cm"],"cm","Named section centre, Figure 7","Figure 7 triangle")
    for r in src("paladio-figure-7.csv"):
        original[r["id"]]=(r["id"],r["reading_cm"],"cm","Figure 7 original graticule",r["mark"])
    for r in src("koyaguchi-figure-3.csv"):
        original[r["id"]]=(r["id"],r["c1_mm"]+" / "+r["c2_mm"],"mm, C1 / C2","C1 map pin checked against C2",r["survey"])
    for r in src("wiesner-wang-table-1.csv"):
        original["WW-"+r["station"]]=(r["station"],r["reading_cm"],"cm","Table 1 latitude and longitude","1994 bed or named trap")
    for r in src("wiesner-2004-table-1.csv"):
        original["W04-"+r["station"]]=(r["station"],r["bulk_mm"],"mm","PANGAEA event "+r["coordinate_event"],"Bulk ash, Table 1")
    for r in src("hess-table-1.csv"):
        original["H01-"+r["site"].zfill(2)]=(r["stations"],r["thickness_mm"] if r["site"]!="14" else "<2 cm in discussion","mm unless stated","Table 1 grouped station coordinates","Core revisits combined")
    for r in src("casadevall-airfields.csv"):
        original["AIR-"+r["station"]]=(r["station"],r["reported_total_cm"],"cm","Named facility reference point","June 15–16 accumulation")
    return original


def archive_inventory():
    """Make location-only posterior checks for the two offshore survey maps."""
    sites = []
    for row in read(AUDIT/"survey-coverage/wiesner-1996-open-symbols.csv"):
        sites.append(dict(id=row["figure_id"], name=row["figure_id"], archive="marine core",
                          latitude=float(row["map_lat"]), longitude=float(row["map_lon"]),
                          reading="inspected_no_bed", confidence="secure", reported_fallout_limit_cm=None,
                          reference="Wiesner and Wang (1996), Figure 1 open symbol",
                          inspection="No visible discrete horizon. Original map centre, no local visibility or preservation calibration."))
    readings = {"filled": "occurrence_only", "open": "inspected_no_bed",
                "partially filled": "unclear_thickness", "overprinted or unresolved": "unresolved_horizon"}
    for i, row in enumerate(read(AUDIT/"survey-coverage/wiesner-2004-station-probes.csv"), 1):
        symbol = row["corrected_fig1_symbol_review"]
        reading = readings[symbol.split(";")[0]]
        sites.append(dict(id=f"W04-probe-{i:02d}", name=row["station"], archive="marine core",
                          latitude=float(row["latitude"]), longitude=float(row["longitude"]), reading=reading,
                          confidence="tentative" if reading == "unresolved_horizon" else "secure",
                          reported_fallout_limit_cm=None, reference="Wiesner et al., corrected Figure 1 and PANGAEA "+row["cast"],
                          inspection=symbol+". Cruise station locator. Reoccupied stations may overlap. Independently tabulated beds enter the quantitative fit once."))
    inventory = {"schema": "tephra-archive-evidence-v1", "target": "Pinatubo 15 June 1991 climactic airborne fallout",
                 "sites": sites, "scenarios": [],
                 "interpretation": "Reported map symbols are prediction locations. No site-specific minimum preservation or visibility calibration was established."}
    (AUDIT/"archive-probes.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False)+"\n")


def build():
    archive_inventory()
    schematic_rows=schematic()
    write(AUDIT/"survey-coverage"/"david-schematic.csv",schematic_rows)
    cards={r["evidence_id"]:r["sheet"] for r in read(AUDIT/"reading-index.csv")}
    originals=transcripts()
    output=[]
    def add(identity,source,locator,raw,unit,lon,lat,georef,kind,value,row,note,verification):
        output.append(dict(evidence_id=identity,source=source,source_record=locator,published_reading=raw,units=unit,
                           longitude=lon,latitude=lat,position_evidence=georef,model_kind=kind,
                           model_value_cm=value,accepted_csv_row=row,decision=note,verification=verification))
    for s in read(AUDIT/"selection.csv"):
        i=s["id"];locator,raw,unit,pos,role=originals[i]
        page=cards.get(i,"marine-check.md" if s["source"] in ("WW1","W04","H01") else "README.md#airfields-and-distal-reports")
        add(i,s["source"],locator,raw,unit,s["longitude"],s["latitude"],pos,s["kind"],s["value_cm"] if s["kind"] else "",
            s["accepted_csv_row"],role+". "+s["decision"],page)
    for r in schematic_rows:
        add("D1-"+r["site"],"D1",r["site"],r["coarse_unit_cm"],"cm",r["approximate_lon"],r["approximate_lat"],
            "Two-city schematic registration; approximate locality only","","","","Coarse unit after six months. Unresolved exact field location; site-level comparison outside the fit.",cards[r["site"]])
    for r in read(AUDIT/"survey-coverage"/"wiesner-1996-open-symbols.csv"):
        add(r["figure_id"],"WW-F1",r["figure_id"],"no visible discrete horizon","",r["map_lon"],r["map_lat"],
            "Original Figure 1 graticule","","","","No site-specific original-bed visibility threshold. Nearby cruise events are candidate locations only.","survey-coverage/wiesner-1996-coverage.png")
    for j,r in enumerate(read(AUDIT/"survey-coverage"/"wiesner-2004-station-probes.csv"),1):
        add(f"W04-probe-{j:02d}","W04-F1",r["station"],r["corrected_fig1_symbol_review"],"",r["longitude"],r["latitude"],
            "PANGAEA cruise event "+r["cast"],"","","","Map probe; use only independently tabulated thickness for fitting. Reoccupied locations overlap.","marine-check.md")
    probes=read(AUDIT/"survey-coverage"/"wiesner-2004-station-probes.csv")
    for r in read(SOURCE/"context/wetzel-2009-sections.csv"):
        hit=next((p for p in probes if p["station"]==r["cruise_station"]),{})
        add("W09-"+r["site"],"W09","Figure "+r["figure"]+" station "+r["site"],r["reported_cm"],"cm",
            hit.get("longitude",""),hit.get("latitude",""),"Cruise station match" if hit else "Station listed; coordinates unassigned",
            "","","",r["description"],"marine-check.md")
    for j,r in enumerate(read(SOURCE/"context/other-field-reports.csv"),1):
        add(f"CTX-{j:02d}","Context",r["source"]+" · "+r["site"],r["reported"],r["unit"],"","","Named context",
            "","","",r["decision"],"README.md")
    for i, r in enumerate(read(SOURCE/"context/paladio-figure-9-localities.csv"), 1):
        add(f"P9-{i:02d}", "P9", r["locality"], r["symbol_reading"], "mapped report", "", "",
            f"Figure 9 pixel ({r['source_x']}, {r['source_y']})", "", "", "", r["qualification"], "distal-reports.md")
    for r in read(SOURCE/"context/smithsonian-field-reports.csv"):
        add(r["evidence_id"], "BGVN", r["location"], r["reported"], r["unit"], "", "",
            "Bulletin "+r["bulletin_issue"]+", "+r["scope"], "", "", "", r["decision"], "distal-reports.md")
    if len({r["evidence_id"] for r in output})!=len(output):
        raise ValueError("Duplicate evidence ID")
    write(AUDIT/"full-evidence.csv",output)


if __name__=="__main__":
    build()
