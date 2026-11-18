"""Eruption-balanced predictive stacking, using only independent calibration eruptions."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .fields import probabilities
from .members import NAMES


PACKET = Path(__file__).resolve().parents[3]/"assets"/"directional-calibration.json"


def solve(log_scores, observation_weights, penalty=1.):
    """The penalty has the weight of one eruption, split evenly among structures."""
    matrix = np.asarray(log_scores, float)
    row_w = np.asarray(observation_weights, float)
    if matrix.ndim != 2 or row_w.shape != (len(matrix),) or not np.isfinite(matrix).all():
        raise ValueError("Invalid calibration scores")
    density = np.exp(matrix-matrix.max(1, keepdims=True))
    count = matrix.shape[1]
    def objective(w):
        mixture = np.maximum(density @ w, 1e-300)
        loss = -row_w @ np.log(mixture)
        grad = -(row_w/mixture) @ density
        if penalty:
            loss -= penalty*np.mean(np.log(w))
            grad -= penalty/(count*w)
        return loss, grad
    res = minimize(objective, np.full(count, 1/count), method="SLSQP", jac=True,
                   bounds=[(1e-9, 1.)]*count,
                   constraints=[{"type":"eq", "fun":lambda w: w.sum()-1., "jac":lambda w: np.ones(count)}],
                   options={"ftol":1e-11, "maxiter":500})
    if not res.success:
        raise ArithmeticError("Stacking did not converge: "+res.message)
    return probabilities(res.x)


def calibrate(meta, slug=None, packet=None):
    packet = json.loads(PACKET.read_text()) if packet is None else packet
    if tuple(packet["members"]) != NAMES:
        raise ValueError("Calibration structures differ from fitting structures")
    included, excluded, rows, scales = [], {}, [], []
    vent = np.asarray(meta["vent_lonlat"])
    for rec in packet["cases"]:
        other = np.asarray(rec["vent_lonlat"])
        same_vent = (abs((other[0]-vent[0]+180) % 360-180) < .01 and abs(other[1]-vent[1]) < .01)
        if slug == rec["slug"] or meta["input_sha256"] == rec["input_sha256"] or same_vent:
            excluded[rec["slug"]] = "target or same recorded source"
            continue
        groups = [g for g in rec["folds"] if g["usable"]]
        if not groups:
            excluded[rec["slug"]] = "no jointly usable geographic fold"
            continue
        included.append(rec["slug"])
        rows.extend(g["log_scores"] for g in groups)
        scales.append(np.concatenate([np.full(len(g["row_ids"]), 1/(len(groups)*len(g["row_ids"]))) for g in groups]))
    uniform = np.full(len(NAMES), 1/len(NAMES))
    if included:
        m = np.vstack(rows)
        w = solve(m, np.concatenate(scales))
        unregularized = solve(m, np.concatenate(scales), penalty=0.)
        generator = np.random.default_rng(20261118)
        bootstrap_weights = [solve(m, np.concatenate([s*x for s,x in zip(scales, draw, strict=True)]))
                for draw in len(scales)*generator.dirichlet(np.ones(len(scales)), size=128)]
        stability = np.quantile(bootstrap_weights, [.05,.95], axis=0).tolist()
    else:
        w = unregularized = uniform
        stability = None
    return {"members":list(NAMES), "weights":w.tolist(), "uniform":uniform.tolist(),
            "unregularized":unregularized.tolist(), "eruption_bootstrap_90pct":stability,
            "calibration_cases":included, "excluded_cases":excluded, "penalty_eruption_equivalents":1.}
