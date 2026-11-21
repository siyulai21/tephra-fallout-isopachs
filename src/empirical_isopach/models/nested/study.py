"""Fit each accepted observation and buffered geographic partition with the same prior."""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from ...core.audit import audited_data
from ...validation import automatic_buffer
from ..directional.ensemble.members import splits
from .inference import Deposit, fit
from .profile import ASSET


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def engine_digest():
    root = Path(__file__).parent
    return sha256(b"".join((root/k).read_bytes() for k in
                  ("geometry.py","profile.py","budget_profile.py","inference.py","study.py"))+ASSET.read_bytes()).hexdigest()


def save_draws(path, model):
    np.savez_compressed(path,z=model.z,geometry=model.geometry,power_tail=model.power_tail,
                        latitude=model.latitude)


def load_draws(path):
    with np.load(path,allow_pickle=False) as f:
        return Deposit(f["z"],f["geometry"],f["power_tail"],float(f["latitude"]))


def infer_dataset(path, output, *, budget=True):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data, meta = audited_data(path,allow_intervals=True)
    lat = meta["vent_lonlat"][1]
    signature = engine_digest()
    folds = []
    for index,train,test in [(None,data,None),*splits(data,lat,"geographic")]:
        key = "full" if index is None else f"fold-{index}"
        packet = output/(key+".npz")
        receipt = output/(key+".json")
        expected = {"protocol":"nested-budget-v2" if budget else "nested-isopach-v1","engine_sha256":signature,
                    "input_sha256":meta["input_sha256"],"training_rows":train.row_ids.tolist(),
                    "test_rows":[] if test is None else test.row_ids.tolist()}
        row = json.loads(receipt.read_text()) if receipt.exists() else {}
        if row.get("identity")!=expected or not packet.exists() or digest(packet)!=row.get("posterior_sha256"):
            model, numerics = fit(train,lat,budget=budget)
            save_draws(packet,model)
            row = {"identity":expected,"posterior_sha256":digest(packet),"numerics":numerics}
            if test is not None:
                scores = model.score(test)
                row["row_log_scores"] = scores.tolist()
                row["negative_log_score"] = float(-scores.mean())
                row["median_log_cm"] = np.median(model.field(test.xy),axis=0).tolist()
            write_json(receipt,row)
        print(key, len(train.xy), row["numerics"]["passed"], flush=True)
        if test is not None:
            folds.append({"fold":index,**row})
    validation = {"buffer_km":automatic_buffer(data)[0],"folds":folds,
                  "equal_region_negative_log_score":float(np.mean([r["negative_log_score"] for r in folds]))}
    write_json(output/"geographic.json",validation)
    return validation


if __name__ == "__main__":
    import argparse
    from ...core.catalog import resolve_dataset
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("output",type=Path)
    a = p.parse_args()
    infer_dataset(resolve_dataset(a.case),a.output)
