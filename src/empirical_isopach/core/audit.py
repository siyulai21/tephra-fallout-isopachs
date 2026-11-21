"""Verify registered inputs and identify their interpreted observation bounds."""
import csv
from hashlib import sha256
import json
from pathlib import Path

from .catalog import data_directory, dataset_record
from .observations import read_observations


def audited_data(path, *, allow_intervals=False):
    data, meta = read_observations(path,allow_intervals=allow_intervals)
    record = dataset_record(path)
    meta["source"] = "data/"+record["file"] if record else Path(path).name
    if not record or not record.get("audit"):
        return data, meta
    audit_path = data_directory()/record["audit"]
    audit = json.loads(audit_path.read_text())
    case = audit["cases"][record["audit_case"]]
    if meta["input_sha256"] != case["sha256"]:
        raise ValueError("Accepted input differs from its recorded audit.")
    meta["audit"] = {"file": record["audit"], "sha256": sha256(audit_path.read_bytes()).hexdigest(),
                     "case": record["audit_case"], "definition": case["definition"]}
    limits = audit.get("principal_limits", []) if record["audit_case"] == "principal" else []
    if limits:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(enumerate(csv.DictReader(stream), start=2))
        interpreted = []
        for limit in limits:
            sid = limit["evidence_id"]
            found = [(i,r) for i,r in rows if r.get("EvidenceID") == sid]
            if len(found) != 1:
                raise ValueError(f"Interpreted upper limit {sid} must identify one observation")
            row_id, row = found[0]
            upper = limit["interpreted_upper_cm"]
            if (row["Kind"] != "upper_limit" or row["Thickness"] or float(row["Limit_cm"]) != upper
                    or row_id not in data.row_ids):
                raise ValueError(f"Interpreted upper limit disagrees with the observation at {sid}")
            interpreted.append({"id":sid, "name":row["Site"], "row_id":row_id, "limit_cm":upper,
                                "latitude":float(row["Lat"]), "longitude":float(row["Lon"])})
        if len({s["row_id"] for s in interpreted}) != len(interpreted):
            raise ValueError("Repeated interpreted constraint")
        meta["interpreted_upper_limits"] = interpreted
    return data, meta
