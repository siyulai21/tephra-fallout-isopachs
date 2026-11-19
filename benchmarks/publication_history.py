"""Run frozen cross-eruption checks against the B-Tm input and case they published."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/"eruptions/b-tm/history/2026-11-18"
MODULES = {"verify_aso_evidence", "verify_directional", "verify_directional_ensemble", "verify_visuals"}


def verify_previous(module, args=()):
    """Use live files for unchanged evidence and verified original bytes for replaced files."""
    if module not in MODULES:
        raise ValueError(module)
    manifest = json.loads((ARCHIVE/"manifest.json").read_text())
    args = list(args)
    if "--output" in args:
        index = args.index("--output") + 1
        args[index] = str(Path(args[index]).resolve())
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    with TemporaryDirectory(prefix="btm-publication-") as tmp:
        view = Path(tmp)
        names = set(manifest["files"])
        for filename in tracked:
            if not filename or filename in names or filename.startswith("eruptions/b-tm/history/"):
                continue
            source = ROOT/filename
            if not source.is_file():
                continue
            target = view/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        for name in sorted(names):
            target = view/name
            if not target.resolve().is_relative_to(view.resolve()):
                raise ValueError(f"Unsafe archive path {name}")
            frozen = ARCHIVE/"files"/(name + ".txt" if name.endswith(".md") else name)
            content = frozen.read_bytes()
            if sha256(content).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Corrupt publication file {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        # The archived verifiers themselves are restored to their published implementations.
        subprocess.run([sys.executable, "-m", "benchmarks."+module, *args], cwd=view, check=True)
