"""Find bundled input data both in a checkout and an installed distribution."""
import json
from pathlib import Path
import sys


def data_directory():
    candidates = [Path(__file__).parents[2]/"data",
                  Path(sys.prefix)/"share/tephra-fallout-isopachs"]
    for candidate in candidates:
        if (candidate/"datasets.json").exists():
            return candidate
    raise FileNotFoundError("Bundled datasets were not found. Run from a repository checkout or reinstall the distribution.")


def datasets():
    root = data_directory()
    return [{**row, "path": root/row["file"]} for row in json.loads((root/"datasets.json").read_text())["datasets"]]


def resolve_dataset(value):
    value = Path(value)
    if value.is_file() or value.suffix.lower() == ".csv" or len(value.parts) > 1:
        return value
    for row in datasets():
        if str(value).lower() in {row["name"].lower(), row["slug"]}:
            return row["path"]
    return value


def dataset_record(path):
    """Match only the actual catalogued file, never a same-named external CSV."""
    resolved = Path(path).resolve()
    return next((row for row in datasets() if row['path'].resolve() == resolved), None)
