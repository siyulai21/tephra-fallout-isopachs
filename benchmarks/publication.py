"""Read and verify the SHA-256 identities of the published evidence."""
from hashlib import sha256
from pathlib import Path

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
