"""Resolve archived presentation edits without relaxing scientific checksums."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT/"docs/maintenance/visuals-2026-11-17.json"
NAVIGATION = ROOT/"docs/maintenance/navigation-2026-11-18.json"


def is_presentation(path):
    path = Path(path)
    return path.suffix in (".png", ".svg") or path.name in ("README.md", "artifacts.json")


def matches(path, expected):
    return path.is_file() and sha256(path.read_bytes()).hexdigest() == expected


def reviewed_destination(path, record):
    target = ROOT/record["current_path"]
    return ((target == path or not path.exists())
            and matches(target, record["current_sha256"]))


def verified_navigation(path, expected):
    """Verify an exact pre-edit publication and the exact page that replaced it."""
    if matches(path, expected):
        return True
    if not path.is_relative_to(ROOT) or not is_presentation(path):
        return False
    key = path.relative_to(ROOT).as_posix()
    release = json.loads(NAVIGATION.read_text())
    move = release["unchanged_moves"].get(key)
    if move:
        return move["current_sha256"] == expected and reviewed_destination(path, move)
    record = release["files"].get(key)
    if path.name not in ("README.md", "artifacts.json"):
        return False
    if not record or record["prior_sha256"] != expected or not reviewed_destination(path, record):
        return False
    with ZipFile(NAVIGATION.with_suffix(".zip")) as previous:
        return sha256(previous.read(key)).hexdigest() == expected


def verify_navigation_release():
    release = json.loads(NAVIGATION.read_text())
    archive = NAVIGATION.with_suffix(".zip")
    assert matches(archive, release["archive_sha256"])
    with ZipFile(archive) as previous:
        assert set(previous.namelist()) == set(release["files"])
        for name, record in release["files"].items():
            assert Path(name).name in ("README.md", "artifacts.json"), name
            assert Path(record["current_path"]).name in ("README.md", "artifacts.json"), name
            assert sha256(previous.read(name)).hexdigest() == record["prior_sha256"], name
            assert reviewed_destination(ROOT/name, record), name
    for name, record in release["unchanged_moves"].items():
        assert is_presentation(name) and is_presentation(record["current_path"])
        assert reviewed_destination(ROOT/name, record), name


def verified_presentation(path, expected_digest):
    """Follow only named revisions, checking both live and archived evidence."""
    path = Path(path).resolve()
    if verified_navigation(path, expected_digest):
        return True
    if not path.is_relative_to(ROOT) or not is_presentation(path):
        return False
    key = path.relative_to(ROOT).as_posix()
    archive = ROOT/"comparisons/directional-ensemble/history/2026-11-18"
    if not (archive/"manifest.json").is_file():
        return False
    record = json.loads((archive/"manifest.json").read_text())["files"].get(key)
    if not record or record["prior_sha256"] != expected_digest:
        return False
    if not verified_navigation(path, record["current_sha256"]):
        return False
    with ZipFile(archive/"previous-presentation.zip") as previous:
        return sha256(previous.read(key)).hexdigest() == expected_digest


def verified_revision(path, original_digest):
    """A frozen hash may advance only through a named presentation revision."""
    path = Path(path)
    if not is_presentation(path) or not LEDGER.is_file():
        return False
    record = json.loads(LEDGER.read_text())["replaced_presentation_files"].get(path.as_posix())
    return bool(record and record["before_sha256"] == original_digest
                and verified_presentation(path, record["after_sha256"]))
