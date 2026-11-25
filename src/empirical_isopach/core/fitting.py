"""Retain reported fallout minima while fitting their numeric thresholds as point values."""
import csv
from hashlib import sha256
import io
from pathlib import Path

from .catalog import data_directory, dataset_record
from .observations import read_observations

MARKER = "Reported_lower_limit_cm"
POLICY = "lower-only-at-reported-minimum-v1"


def fitting_bytes(source):
    """Return canonical fitting bytes, or the original bytes if no row changes."""
    raw = Path(source).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    fields = reader.fieldnames or []
    changed = []
    for row in rows:
        pseudo = (row.get("Pseudo") or "").strip().lower()
        if pseudo not in {"yes", "true", "1", "1.0"} and (row.get("Kind") or "").strip().lower() == "lower_limit":
            changed.append(row)
    if not changed:
        return raw
    # Reject invalid source intervals and limits before translating their evidence.
    read_observations(source)
    if MARKER in fields and any(r.get(MARKER) for r in changed):
        raise ValueError("A reported lower-only row already carries a fitting marker.")
    for row in changed:
        row[MARKER] = row["Limit_cm"]
        row["Kind"], row["Thickness"], row["Limit_cm"] = "measurement", row["Limit_cm"], ""
    result = io.StringIO(newline="")
    writer = csv.DictWriter(result, fieldnames=fields if MARKER in fields else [*fields, MARKER], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return result.getvalue().encode()


def model_input(path, output):
    """Resolve a verified fitting table without editing any source CSV."""
    path = Path(path)
    record = dataset_record(path)
    if record and record.get("reported_file"):
        return path
    effective = fitting_bytes(path)
    original = path.read_bytes()
    if effective == original:
        return path
    if record:
        from .catalog import datasets
        current = next(r for r in datasets() if r["slug"] == record["slug"])
        if current.get("reported_file") and (data_directory() / current["reported_file"]).resolve() == path.resolve():
            if current["path"].read_bytes() != effective:
                raise ValueError("Registered fitting input differs from the reported limits.")
            return current["path"]
    # Content addressing keeps old and new fits apart when an external CSV is updated.
    directory = Path(output).parent / "fitting-inputs" / sha256(original).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    if target.exists() and target.read_bytes() != effective:
        raise ValueError(f"Existing fitting input has different contents: {target}")
    if not target.exists():
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(dir=directory, delete=False) as tmp:
            tmp.write(effective)
        Path(tmp.name).replace(target)
    return target
