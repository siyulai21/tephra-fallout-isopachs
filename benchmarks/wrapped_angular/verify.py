"""Replay published field, input, contour and paired-validation identities."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.wrapped_angular.packet import FAMILY, PROTOCOL, read
from src.empirical_isopach.models.wrapped_angular.validation import partitions
from src.empirical_isopach.models.wrapped_angular.fitting import source_for
from benchmarks.publication import result_home
from src.empirical_isopach.models.wrapped_angular.report import engine_sha256, verified_release

ROOT = Path("comparisons/wrapped-angular-covariance")


def verify(record, peers):
    data, meta = audited_data(record["path"])
    home = result_home(record) / FAMILY
    manifest = json.loads((home / "artifacts.json").read_text())
    assert manifest["identity"]["engine_sha256"] == engine_sha256() and verified_release(
        home, manifest["identity"]
    )
    rec = json.loads((home / "report.json").read_text())
    assert rec["protocol"] == PROTOCOL and rec["input_sha256"] == meta["input_sha256"]
    field = read(home / "posterior.npz", meta)
    assert rec["source_area"] == source_for(meta, data)
    assert set(rec["fit"]["source_rows"]).isdisjoint(rec["fit"]["exterior_rows"])
    assert sorted(rec["fit"]["source_rows"] + rec["fit"]["exterior_rows"]) == sorted(data.row_ids)
    full = pd.read_csv(home / "measured-sites.csv.gz").set_index("row_id").loc[data.row_ids]
    probes = np.linspace(0, len(data.xy) - 1, 13, dtype=int)
    np.testing.assert_allclose(
        np.exp(field.quantiles(data.xy[probes], (0.5,))[0]),
        full.primary_median_cm.to_numpy()[probes],
        rtol=2.0e-6,
    )
    saved = json.loads((home / "partitions.json").read_text())
    events = pd.read_csv(home / "heldout.csv.gz")
    for design in ("geographic", "quadrants", "distal"):
        for k, train, test in partitions(data, meta, record["path"], design):
            if train is None or not len(test.xy) or not len(train.xy):
                continue
            block = saved[f"{design}-{k}"]
            assert block["train"] == train.row_ids.tolist() and block["test"] == test.row_ids.tolist()
            native = (
                events[(events.design == design) & (events.fold == k)].set_index("row_id").loc[test.row_ids]
            )
            common = peers[
                (peers.design == ("radial" if design == "distal" else design))
                & (peers.fold == k)
                & (peers.family == FAMILY)
            ]
            old = common.set_index("row_id").loc[test.row_ids]
            assert (old.input_sha256 == meta["input_sha256"]).all()
            np.testing.assert_allclose(-native.log_score, old.nll, atol=2.0e-7, rtol=2.0e-7)
            np.testing.assert_allclose(native.cdf, old.cdf, atol=2.0e-7)
            if design == "geographic":
                for other in ("directional", "spherical-mixture", "nested-isopachs"):
                    block = peers[(peers.family == other) & (peers.design == design) & (peers.fold == k)]
                    np.testing.assert_array_equal(np.sort(block.row_id), np.sort(test.row_ids))
                    assert (block.input_sha256 == meta["input_sha256"]).all()
    audit = json.loads(gzip.decompress((home / "support.json.gz").read_bytes()))
    assert audit["policies"]
    result = json.loads((home / "deposit.json").read_text())
    earth = json.loads((home / "earth-integrals.json").read_text())
    np.testing.assert_allclose(earth["volumes_km3"], result["thickness_band"]["volumes_km3"], rtol=0.015)
    assert earth["passed"] and result["integration_passed"] and result["thickness_band"]["passed"]
    print(record["slug"], len(data.xy), f"{result['median_map_volume_km3']:.4g} km³", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    args = parser.parse_args()
    comparison = pd.read_csv(ROOT / "events.csv.gz")
    for rec in datasets():
        if args.case is None or rec["slug"] == args.case:
            verify(rec, comparison[comparison.slug == rec["slug"]])


if __name__ == "__main__":
    main()
