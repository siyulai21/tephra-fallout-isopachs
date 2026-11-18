"""Resolve explicitly reviewed figure revisions without exempting science files."""
from hashlib import sha256
import json
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1]/"docs/maintenance/visuals-2026-11-17.json"


def is_presentation(path):
    path = Path(path)
    return path.suffix in (".png", ".svg") or path.name in ("README.md", "artifacts.json")


def verified_revision(path, original_digest):
    """A frozen hash may advance only through a named presentation revision."""
    path = Path(path)
    if not is_presentation(path) or not LEDGER.is_file():
        return False
    record = json.loads(LEDGER.read_text())["replaced_presentation_files"].get(path.as_posix())
    return bool(record and record["before_sha256"] == original_digest
                and record["after_sha256"] == sha256(path.read_bytes()).hexdigest())
