"""Resolve explicitly reviewed figure revisions without exempting science files."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

LEDGER = Path(__file__).resolve().parents[1]/"docs/maintenance/visuals-2026-11-17.json"


def is_presentation(path):
    path = Path(path)
    return path.suffix in (".png", ".svg") or path.name in ("README.md", "artifacts.json")


def verified_presentation(path, expected_digest):
    """Accept a dated replacement only if both the current and archived bytes verify."""
    path = Path(path)
    if sha256(path.read_bytes()).hexdigest() == expected_digest:
        return True
    if not is_presentation(path):
        return False
    root = Path(__file__).resolve().parents[1]
    try:
        key = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    archive = root/"comparisons/directional-ensemble/history/2026-11-18"
    if not (archive/"manifest.json").is_file():
        return False
    record = json.loads((archive/"manifest.json").read_text())["files"].get(key)
    if not record or record["prior_sha256"] != expected_digest or record["current_sha256"] != sha256(path.read_bytes()).hexdigest():
        return False
    with ZipFile(archive/"previous-presentation.zip") as z:
        return sha256(z.read(key)).hexdigest() == expected_digest


def verified_revision(path, original_digest):
    """A frozen hash may advance only through a named presentation revision."""
    path = Path(path)
    if not is_presentation(path) or not LEDGER.is_file():
        return False
    record = json.loads(LEDGER.read_text())["replaced_presentation_files"].get(path.as_posix())
    return bool(record and record["before_sha256"] == original_digest
                and verified_presentation(path, record["after_sha256"]))
