"""Join station logs to archival map probes without duplicating a likelihood row."""
import gzip
import json
import re

from .pinatubo_offshore import AUDIT, REV, SRC, INPUT, read, write


def inventory():
    parent = json.loads((AUDIT/"archive-probes.json").read_text())
    rows = {r["EvidenceID"]:r for r in read(INPUT)}
    old_marks = read(AUDIT/"survey-coverage/wiesner-1996-open-symbols.csv")
    marks = {r["figure_id"]:r for r in old_marks}
    with gzip.open(SRC/"context/so95-events.html.gz","rt") as f:
        raw_events = re.findall(r"<tr[^>]*>(.*?)</tr>", f.read(), re.S)
    events = {}
    for tr in raw_events:
        v = re.findall(r"<td[^>]*>(.*?)</td>",tr,re.S)
        if len(v)>8 and v[3]=="GKG":
            events.setdefault(v[0][3:8],[]).append(v)
    review = []
    for site in parent["sites"]:
        site["original_id"] = site["id"]
        marker = marks.get(site["id"])
        if marker:
            nearby=marker["nearby_box_core_events_within_15_km"]
            site["inspection"] += " Nearby registered casts (possible matches only): "+(nearby or "none within 15 km")+"."
        elif site["name"] in {"SO132_24","SO132_33","SO132_47"}:
            key="W09-"+site["name"].split("_")[1]
            site["id"]=key
            site["in_principal_as"]={"W09-24":"interpreted_lower_limit","W09-33":"interpreted_measurement","W09-47":"interpreted_measurement"}[key]
            site["inspection"]+=" Wetzel's named X-ray section resolves the primary ash. See principal-decisions.csv."
            if key=="W09-47":
                site["reading"]="dispersed_no_continuous_bed"
        review.append(dict(id=site["id"],original_id=site["original_id"],site=site["name"],reading=site["reading"],lon=site["longitude"],lat=site["latitude"],reference=site["reference"],decision=site["inspection"]))
    core = rows["W08-404"]
    observed = dict(id=core["EvidenceID"], name=core["Site"], archive="marine box core",
                    latitude=float(core["Lat"]), longitude=float(core["Lon"]), reading="dispersed_no_continuous_bed",
                    confidence="secure", reported_fallout_limit_cm=None, in_principal_as="interpreted_measurement",
                    reference="Wetzel (2008), Figs. 2 and 5B; Hess Table 1 group 11",
                    inspection="Original 5 cm ash completely mixed by Scolicia; reconstructed point. Separate from the 0–8 cm X-ray window.")
    parent["sites"].append(observed)
    review.append(dict(id=observed["id"], original_id="H01-11", site=core["Site"], reading=observed["reading"],
                       lon=core["Lon"], lat=core["Lat"], reference=observed["reference"], decision=observed["inspection"]))
    for row in read(REV/"so95-surface.csv"):
        st=row["station"]
        key="SO95-"+st
        casts=events.get(st,[])
        selected=[v for v in casts if "WW-"+v[0][3:] in rows]
        if len(casts)>1 and not selected:
            # Where several casts exist and no original quantitative match identifies one,
            # retain the group and use its first registered deployment as a locator.
            selected=casts[:1]
        for event in selected or casts[:1]:
            ident=key if len(selected)<2 else key+"-"+event[0].split("-")[-1]
            status,note="inspected_no_bed","Dash in the inspected ash column."
            if row["ash"]!="dash":
                status,note="occurrence_only","Shipboard reading: "+row["ash"]+". Check the focused tephra analysis for the fitted value."
            elif "WW-"+event[0][3:] in rows:
                status,note="occurrence_only","Shipboard dash conflicts with targeted ash study's positive observation; retain the focused primary reading."
            note+=" Water: "+row["water"]+". Surface: "+row["surface"]+"."
            s=dict(id=ident,name=event[0],archive="marine box core",latitude=float(event[6]),longitude=float(event[7]),
                   reading=status,confidence="secure",reported_fallout_limit_cm=None,reference="Sarnthein et al. 1994, sec. 3.4, surface Table 1; PANGAEA SO95 "+event[0],inspection=note)
            if key in rows:
                s["in_principal_as"]="interpreted_upper_limit"
                s["inspection"]+=" An independently logged underlying homogeneous unit supports a conditional inventory ceiling."
            elif status=="inspected_no_bed":
                s["inspection"]+=" No secure local retention calibration for original fallout."
            parent["sites"].append(s)
            review.append(dict(id=ident,original_id="",site=s["name"],reading=status,lon=s["longitude"],lat=s["latitude"],reference=s["reference"],decision=s["inspection"]))
        if not casts:
            review.append(dict(id=key,original_id="",site="GIK"+st,reading=row["ash"],lon="",lat="",reference="SO95 surface Table 1",decision="No unique position in the archived box-corer register. Retain the shipboard entry."))
    # Link the additional X-ray sections to any existing survey station. Two
    # named SO114 casts have a usable Hess location but no earlier survey query.
    illustrated=read(REV/"wetzel-2008-sections.csv")
    by_station={s["name"]:s for s in parent["sites"]}
    prior={r["evidence_id"]:r for r in read(AUDIT/"full-evidence.csv")}
    for record in illustrated:
        station=record["cruise_station"]
        source="Wetzel (2008), Fig. "+record["figure"]
        if station in by_station:
            matched=by_station[station]
            matched["inspection"] += " "+source+". "+record["field_decision"]
            next(r for r in review if r["id"]==matched["id"])["decision"]=matched["inspection"]
        elif record["site"] in {"402", "408"}:
            locator=prior[{"402":"H01-09","408":"H01-12"}[record["site"]]]
            site=dict(id="W08-"+record["site"], name=station, archive="marine box core",
                      latitude=float(locator["latitude"]), longitude=float(locator["longitude"]),
                      reading="occurrence_only", confidence="secure", reported_fallout_limit_cm=None,
                      reference=source+"; Hess Table 1", inspection=record["field_decision"])
            parent["sites"].append(site)
            review.append(dict(id=site["id"], original_id=locator["evidence_id"], site=station,
                               reading=site["reading"], lon=site["longitude"], lat=site["latitude"],
                               reference=site["reference"], decision=site["inspection"]))
    (REV/"archive-probes.json").write_text(json.dumps(parent,indent=2)+"\n")
    write(REV/"marine-site-decisions.csv",review)
    return parent

if __name__=="__main__":
    inventory()
