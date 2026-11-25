"""Read and verify the SHA-256 identities of the published evidence."""
from functools import lru_cache
from hashlib import sha256
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "benchmarks/published-evidence.sha256"
BTM_THICKNESS_ONLY = Path("archive/2026-11-19-b-tm-thickness-only")


INTERVAL_PREDECESSOR = ROOT / "archive/2026-11-23-before-interval-likelihood"


def published_nested_engine():
    """Reconstruct the pre-interval engine identity from its original source bytes."""
    from src.empirical_isopach.models.nested.profile import ASSET
    root = ROOT / "src/empirical_isopach/models/nested"
    files = [root/name for name in ("geometry.py", "profile.py", "budget_profile.py")]
    files += [INTERVAL_PREDECESSOR/"models/nested"/name for name in ("inference.py", "study.py")]
    engine = sha256(b"".join(path.read_bytes() for path in files) + ASSET.read_bytes()).hexdigest()
    continuation = sha256((ROOT/"archive/2026-11-23-nested-geography/continuation.py").read_bytes()).hexdigest()
    return sha256((engine + continuation).encode()).hexdigest()


def published_reach_engine():
    root = ROOT / "src/empirical_isopach/models/nested"
    components = (sha256((root/name).read_bytes()).hexdigest() for name in ("reach.py", "reach_fit.py"))
    return sha256((published_nested_engine() + "".join(components)).encode()).hexdigest()


def result_home(record):
    """Results root for a current case or its matched five-family study edition."""
    return Path(record.get("results_directory") or record.get("matched_results") or f"eruptions/{record['slug']}/results")


def reported_edition(record):
    """Input and results for the most recent edition using geological censoring."""
    from src.empirical_isopach.core.catalog import data_directory, dataset_record
    source = record.get("reported_file")
    return dataset_record(data_directory() / source) if source else record


def published_methods(record):
    """Families reviewed for this input. An undeclared partial release is an error."""
    from src.empirical_isopach.models.registry import active_methods
    from src.empirical_isopach.visualization.deposit import FAMILIES
    return tuple(record.get("published_methods", (*active_methods(), *FAMILIES)))


def matched_edition(record):
    """Return the input used by the published five-family development studies."""
    from src.empirical_isopach.core.catalog import data_directory, dataset_record
    earlier = [item for item in record.get("previous_inputs", []) if "matched_results" in item]
    if len(earlier) > 1:
        raise ValueError(f"Multiple matched study editions for {record['slug']}")
    return dataset_record(data_directory() / earlier[0]["file"]) if earlier else record


def published_datasets():
    """Verify the actual input of every fixed five-family benchmark before replay."""
    from src.empirical_isopach.core.catalog import datasets
    records = [matched_edition(r) for r in datasets()]
    for r in records:
        report = json.loads((ROOT / result_home(r) / "report.json").read_text())
        if report["input"]["input_sha256"] != sha256(r["path"].read_bytes()).hexdigest():
            raise ValueError(f"Matched study and registered input differ for {r['slug']}")
    return records


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
    relative = Path(name).relative_to("eruptions/mazama/results")
    content = (folder / "controls" / relative).read_bytes()
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


def historical_volume_rows(home):
    """Replay the whole-draw intervals used by the dated source-sensitivity studies."""
    from src.empirical_isopach.visualization.deposit import FAMILIES, read_json, volume_interval
    from src.empirical_isopach.visualization.summary import metrics, principal_report
    home = Path(home)
    regional = read_json(home/"report.json")
    directional = read_json(home/"directional/azimuth/report.json")
    spherical = read_json(home/"spherical-mixture/report.json")
    nested = principal_report(home)
    values = []
    for key,record in regional["methods"].items():
        vol = record["regional_volume"]
        q = record["joint_volume"]["conditional_quantiles_km3"]
        hist = [v["median_surface_km3"] for v in vol["refinement"]]
        ok = len(hist)>=3 and all(abs(a/b-1)<vol["tolerance"] for a,b in zip(hist[-3:-1],hist[-2:]))
        values.append((key,vol["median_surface_km3"],(q["0.025"],q["0.975"]),ok,record["joint_volume"]["expected_grid_converged"]))
    for key,source in zip(FAMILIES,(directional,spherical,nested),strict=True):
        r = read_json(home/key/"deposit.json")
        ext = home/key/"extent/report.json"
        q,ok = volume_interval(key,source,read_json(ext) if ext.is_file() else None)
        values.append((key,r["median_map_volume_km3"],q,r["integration_passed"],ok))
    return [dict(method=k, median_map_volume_km3=v,conditional_volume_lower95_km3=q[0],conditional_volume_upper95_km3=q[1],
                 map_integration_passed=m,complete_draw_checks_passed=passed,heldout_log_score=score,
                 heldout_coverage95=cov,heldout_checks_passed=valid,input_sha256=regional["input"]["input_sha256"])
            for (k,v,q,m,passed),(_,score,cov,valid) in zip(values,metrics(regional,directional,spherical,nested,home=home),strict=True)]
