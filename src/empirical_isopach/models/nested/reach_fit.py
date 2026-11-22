"""Verified training packets and independent-proposal fallback for corridor reach."""
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from .continuation import PROTOCOL as PARENT, check_and_refine, continuation_digest, join, load_packet
from .inference import fit
from .reach import PROTOCOL, ReachDraws, condition
from .study import digest, write_json


def signature():
    root = Path(__file__).parent
    value = continuation_digest() + "".join(digest(root / f) for f in ("reach.py", "reach_fit.py"))
    return sha256(value.encode()).hexdigest()


def conditioned_fit(train, test, meta, base, out, key, seed):
    out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((base / (key + ".json")).read_text())
    source = base / (key + ".npz")
    source_id = receipt["identity"]
    test_ids = [] if test is None else test.row_ids.tolist()
    if (source_id["protocol"] != PARENT or source_id["engine_sha256"] != continuation_digest()
            or source_id["input_sha256"] != meta["input_sha256"] or source_id["training_rows"] != train.row_ids.tolist()
            or source_id["test_rows"] != test_ids or digest(source) != receipt["posterior_sha256"]):
        raise ValueError(f"Input, training partition or posterior differs from {source}")
    identity = {"protocol":PROTOCOL,"engine_sha256":signature(),"input_sha256":meta["input_sha256"],
                "seed":seed,"training_rows":train.row_ids.tolist(),"test_rows":test_ids,
                "parent_posterior_sha256":receipt["posterior_sha256"]}
    file = out / (key + ".npz")
    info = file.with_suffix(".json")
    if info.exists():
        row = json.loads(info.read_text())
        if row["identity"] != identity or digest(file) != row["posterior_sha256"]:
            raise ValueError(f"Saved corridor fit differs from {identity['protocol']} inputs. Use a fresh destination.")
        return ReachDraws.load(file), row
    base_model = load_packet(source)
    model, importance = condition(base_model, train, seed=seed)
    attempts = [importance]
    parent_passed = receipt["numerics"]["passed"]
    if not importance["passed"]:
        positive = train.subset(~train.censored | train.lower_censored)
        # All bounds are reintroduced in the new likelihood. Starting from the
        # positive-only posterior restores shapes suppressed by the old upper-limit fit.
        raw, info_fit = fit(positive, base_model.latitude, initial=768, ceiling=12288, seed=seed)
        raw, proposal, diagnostics = check_and_refine(raw, positive, info_fit, seed=seed, ceiling=12288)
        parent_passed = diagnostics["passed"]
        model, importance = condition(proposal, train, positive_proposal=True, seed=seed)
        attempts.append(importance)
        source_fit = {"training_rows":positive.row_ids.tolist(),"passed":bool(parent_passed),
                      "attempts":[{k:v for k,v in a.items() if k not in ("tempering","log_evidence_exponential_power")}
                                  for a in diagnostics["attempts"]],"continuation_checks":diagnostics["continued_attempts"]}
    else:
        source_fit = None
    model.save(file)
    row = {"identity":identity, "posterior_sha256":digest(file), "importance_attempts":attempts,
           "source_fit_passed":bool(parent_passed), "positive_only_refit":source_fit,
           "passed":bool(parent_passed and importance["passed"])}
    if test is not None:
        row.update(row_log_scores=model.score(test).tolist(), reference_row_log_scores=base_model.score(test).tolist(),
                   latent_median_log_cm=np.median(model.field(test.xy), axis=0).tolist())
    write_json(info, row)
    return model, row
