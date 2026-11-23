"""Read and verify the SHA-256 identities of the published evidence."""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import tarfile
import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "benchmarks/published-evidence.sha256"
BTM_THICKNESS_ONLY = Path("archive/2026-11-19-b-tm-thickness-only")


def previous_btm_file(name):
    """Translate the signed 68-row release's original paths to its present archive."""
    path = Path(name)
    if path.is_relative_to("eruptions/b-tm"):
        return BTM_THICKNESS_ONLY / path.relative_to("eruptions/b-tm")
    if path == Path("data/audits/B-Tm/README.md"):
        return path.with_name("thickness-only.md")
    return path


MAZAMA_ARCHIVE = Path("archive/2026-11-23-mazama-source-audit")


@lru_cache(maxsize=4)
def previous_mazama_bytes(name, root=ROOT):
    """Read an original study artifact from the verified 45-observation release."""
    folder = root / MAZAMA_ARCHIVE
    name = Path(name).as_posix()
    fingerprints = json.loads((folder / "files.json").read_text())
    expected = fingerprints[name]
    manifest = json.loads((folder / "previous-release.tar.gz.parts.json").read_text())
    archive = BytesIO()
    for part in manifest["parts"]:
        content = (folder / part["file"]).read_bytes()
        if len(content) != part["bytes"] or sha256(content).hexdigest() != part["sha256"]:
            raise ValueError(f"Changed Mazama archive part {part['file']}")
        archive.write(content)
    if sha256(archive.getbuffer()).hexdigest() != manifest["original_sha256"]:
        raise ValueError("Changed Mazama release archive")
    archive.seek(0)
    with tarfile.open(fileobj=archive, mode="r:gz") as source:
        content = source.extractfile(name).read()
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"Archived Mazama evidence differs at {name}")
    return content


SPHERICAL_REFERENCE = Path("archive/2026-11-20-spherical-continuation")


def preceding_spherical_bytes(name, root=ROOT):
    """Read a broad-continuation artifact used by a current audit or paired map."""
    path = Path(name)
    folder = root / SPHERICAL_REFERENCE
    archived = folder / "audit-inputs" / path.relative_to("eruptions")
    if path.match("eruptions/*/results/spherical-mixture/report.json") and not archived.exists():
        parent = path.parent / "bearing/parent-report.json"
        if path.is_relative_to("eruptions/mazama"):
            return previous_mazama_bytes(parent, root)
        return (root / parent).read_bytes()
    if path.suffix == ".md":
        with ZipFile(folder / "audit-pages.zip") as pages:
            return pages.read(path.as_posix())
    return archived.read_bytes()


def read_hashes(path=INVENTORY):
    hashes = {}
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in hashes:
            raise ValueError(f"Duplicate publication path: {name}")
        hashes[name] = digest
    return hashes


def verify_files(hashes, root=ROOT):
    for name, expected in hashes.items():
        path = root / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Published evidence is missing or changed: {name}")


# The signed source study predates the bearing revisions. Its originals live at this Git
# revision. The publication inventory authenticates today's outputs independently.
SIGNED_STUDY_REVISION = "843a2eef0fc669e38d95e4153ac8d7464f4d4a98"


@lru_cache(maxsize=1)
def signed_revision_available():
    return subprocess.run(["git", "cat-file", "-e", SIGNED_STUDY_REVISION + "^{commit}"],
                          cwd=ROOT, capture_output=True).returncode == 0


def signed_file(name, expected, current):
    """Check a signed original and independently verify its current replacement, when present."""
    path = ROOT / name
    if path.is_file() and sha256(path.read_bytes()).hexdigest() == expected:
        return True
    if name in current:
        verify_files({name: current[name]})
    if not signed_revision_available():
        return False
    source = subprocess.check_output(["git", "cat-file", "blob", f"{SIGNED_STUDY_REVISION}:{name}"], cwd=ROOT)
    if sha256(source).hexdigest() != expected:
        raise ValueError(f"Signed source differs at {name}")
    return True
