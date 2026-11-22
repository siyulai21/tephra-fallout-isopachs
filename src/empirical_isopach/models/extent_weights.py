"""Transfer weights inferred from buffered geographic deletions in other eruptions."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ASSET = Path(__file__).resolve().parents[1]/"assets"/"extent-calibration.json"


def calibration(family, input_hash, *, exclude_slug=None, ledger=None):
    states = ("broad","resolved")
    asset = json.loads(ASSET.read_text()) if ledger is None else ledger
    donors = [r for r in asset[family] if r["input_sha256"]!=input_hash and r["slug"]!=exclude_slug]
    # Each eligible eruption supplies equal weight. Within eruption, each passed block
    # contributes equally and within block each physical observation contributes equally.
    blocks = []
    cases = []
    for case in donors:
        folds = [f for f in case["folds"] if f["passed"]]
        if not folds:
            continue
        cases.append(case["slug"])
        for fold in folds:
            a = np.array([fold[s] for s in states])
            blocks.append((a,4./(len(folds)*a.shape[1])))
    def loss(theta):
        log_w = np.r_[theta,0.]
        log_w -= logsumexp(log_w)
        # Symmetric Dirichlet(2,...) regularization, one log-weight per structural choice.
        return -sum(c*logsumexp(a+log_w[:,None],axis=0).sum() for a,c in blocks)-log_w.sum()
    result = minimize(loss,np.zeros(len(states)-1),method="BFGS",tol=1e-6)
    if np.linalg.norm(result.jac)>1e-4:
        raise ArithmeticError("Transfer weights did not converge")
    logits = np.r_[result.x,0.]
    weights = np.exp(logits-logsumexp(logits))
    return dict(states=list(states),weights=weights.tolist(),donors=cases,eligible_folds=len(blocks))
