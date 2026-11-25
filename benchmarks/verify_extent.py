"""Check the matched continuation study against case data, saved fields and row-level predictions."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .publication import published_datasets as datasets, result_home
from .footprint_study import controls
from src.empirical_isopach.models.extent_release import draw_check,predictive_summary
from src.empirical_isopach.models.extent_review import FAMILY_NAMES
from src.empirical_isopach.models.extent_weights import ASSET,calibration
from src.empirical_isopach.visualization.deposit import digest,read_json


def main():
    table=pd.read_csv("comparisons/global-extent/paired-results.csv").set_index(["slug","method"])
    kernels=read_json("comparisons/global-extent/kernel-integrals.json")
    sectors=pd.read_csv("comparisons/global-extent/sector-areas.csv.gz")
    angular=pd.read_csv("comparisons/global-extent/angular-variation.csv")
    ledger=read_json(ASSET)
    for rec in datasets():
        slug=rec["slug"]
        home=result_home(rec)
        regional=read_json(home/"report.json")
        partitions=regional["validation"]["methods"]["stacked"]["selections"]
        for family in FAMILY_NAMES:
            folder, map_home = controls(slug,rec)[:2] if family=="nested-isopachs" else (home/family,home/family)
            ext = folder/"extent"
            r = read_json(ext/"report.json")
            print(slug,family,flush=True)
            assert r["input_sha256"]==digest(rec["path"]) and r["calibration_sha256"]==digest(ASSET)
            assert r["heldout_sha256"]==digest(ext/"heldout.csv.gz") and r["volume_draws_sha256"]==digest(ext/"volumes.npz")
            assert all(digest(folder/k)==v for k,v in r["parent_sha256"].items())
            cal=calibration(family,r["input_sha256"],exclude_slug=slug)
            assert cal["donors"]==r["calibration"]["donors"] and slug not in cal["donors"]
            np.testing.assert_allclose(cal["weights"],r["calibration"]["weights"],atol=1e-5)
            frame=pd.read_csv(ext/"heldout.csv.gz")
            rebuilt,metrics=predictive_summary(frame,r["folds"],r["calibration"])
            np.testing.assert_allclose(rebuilt.principal_log_score,frame.principal_log_score,atol=1e-10)
            for kind,v in metrics.items():
                np.testing.assert_allclose(v["fold_scores"],r["validation"][kind]["fold_scores"],atol=1e-10)
                assert v["all_folds_passed"]==r["validation"][kind]["all_folds_passed"]
            study=next((rec for rec in ledger[family] if rec["input_sha256"]==r["input_sha256"]),None)
            for f in r["folds"]:
                if f["status"]!="estimated":continue
                assert not set(f["train"]) & set(f["test"])
                assert f["maximum_training_log_cm_change"]<1e-8
                if f["design"]!="geographic":continue
                base=partitions[f["fold"]]
                assert f["train"]==base["training_row_ids"] and f["test"]==base["withheld_row_ids"]
                rows=frame[(frame.design=="geographic")&(frame.fold==f["fold"])]
                assert rows.row_id.tolist()==f["test"]
                if study is not None:
                    block=next(b for b in study["folds"] if f["fold"]==b["i"])
                    for s in r["state_order"]:
                        np.testing.assert_allclose(rows[s+"_log_score"]-rows.parent_log_score,block[s],atol=6e-10)
            with np.load(ext/"volumes.npz") as a:
                q,error=draw_check(a["volume_km3"],a["weights"],r["calibration"])
                assert np.isfinite(a["volume_km3"]).all() and np.min(a["volume_km3"])>0
            if not r["exact_parent_identity"]:
                np.testing.assert_allclose(q[[0,2]],r["conditional_interval95_km3"],rtol=1e-5)
            audit=json.loads(gzip.decompress((ext/"support.json.gz").read_bytes()))
            assert audit["status"]!="completed" or len(audit["refined_levels"])>=25
            pair=table.loc[(slug,family)]
            assert pair.input_sha256 == r["input_sha256"] and pair.accepted == len(regional["validation"]["row_ids"])
            sector=sectors[(sectors.slug==slug) & (sectors.method==family)]
            assert len(sector)==32 and set(sector.input_sha256)=={pair.input_sha256}
            shape=angular[(angular.slug==slug) & (angular.family==family)]
            if not shape.empty:
                assert set(shape.input_sha256)=={pair.input_sha256}
            receipt = read_json(map_home/"deposit.json")
            np.testing.assert_allclose([pair.principal_km3,pair.lower95,pair.upper95,pair.principal_negative_log_score],
                                       [receipt["median_map_volume_km3"],*r["conditional_interval95_km3"],
                                        r["validation"]["geographic"]["negative_log_score"]])
            if family=="spherical-mixture":
                priors=read_json(ext/"prior-sensitivity.json")
                fits=read_json(folder/"report.json")["fits"]
                assert set(priors)==set(fits)
                for key,v in priors.items():
                    assert v["posterior_sha256"]==fits[key]["posterior_sha256"]
                check = kernels[slug]
                if "input_sha256" in check:
                    assert check["input_sha256"] == r["input_sha256"]
                    assert check["posterior_sha256"] == digest(folder/"posterior.npz")
                    if not check.get("exact_parent_identity"):
                        last = check["attempts"][-1]
                        assert check["passed"] == (last["relative_quantile_difference"] < .08
                                                   and last["relative_scramble_spread"] < .05)
                else:
                    assert check.get("exact_parent_identity") or check["passed"]
    print(f"Verified {len(table)} complete-deposit comparisons and their independent checks.")


if __name__=="__main__":
    main()
