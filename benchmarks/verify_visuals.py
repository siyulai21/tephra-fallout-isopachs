"""Verify published display revisions and untouched scientific evidence."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.core.catalog import datasets
from .visual_history import LEDGER, is_presentation


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path(__file__).resolve().parents[1]
    ledger = read(LEDGER)
    for path, expected in ledger["preserved_scientific_files"].items():
        assert digest(root/path) == expected, path
    for path, revision in ledger["replaced_presentation_files"].items():
        assert is_presentation(path), path
        assert digest(root/path) == revision["after_sha256"], path
    for path, expected in ledger["added_visual_artifacts"].items():
        assert digest(root/path) == expected, path
    for record in datasets():
        home = root/"eruptions"/record["slug"]/"results"
        report = read(home/"report.json")
        replay = read(home/"figure-replay.json")
        for path, expected in replay["source_reports"].items():
            assert digest(home/path) == expected
        for check in replay["regional_parity"].values():
            assert check["recorded_parameters_replayed"] and not check["selection_repeated"]
            assert check["expected_volume_relative_difference"] < 2e-7
            assert check["median_surface_volume_relative_difference"] < 2e-7
        regional = read(home/"map-display.json")
        assert regional["entire_observation_hull_in_frame"] and regional["all_accepted_locations_in_frame"]
        assert regional["unchanged_fit_bounds_km"] == report["fit_bounds_km"]
        assert not regional["volume_calculation_changed"]
        for family in ("directional", "spherical-mixture"):
            folder = home/family
            for path, expected in read(folder/"artifacts.json").items():
                assert digest(folder/path) == expected, str(folder/path)
            display = read(folder/"map-display.json")
            assert display["input_sha256"] == report["input"]["input_sha256"]
            assert display["posterior_sha256"] == digest(folder/"posterior.npz")
            assert not display["volume_calculation_changed"]
            assert display["extent_audit"]["edge_refinements"][-1]["bounds_km"] == display["global_bounds_km"]
            assert np.isfinite(display["global_bounds_km"]).all()
    print(f"Verified {len(ledger['preserved_scientific_files'])} unchanged scientific files, "
          f"{len(ledger['replaced_presentation_files'])} presentation revisions and six complete case visual packets.")


if __name__ == "__main__":
    main()
