"""Holdout-calibrated continuation weights, excluding the eruption under study."""
import json
from pathlib import Path

import numpy as np

from .bearing_support import RATIOS
from .ensemble.members import NAMES
from .ensemble.weights import solve

CANDIDATES = tuple((name, r) for name in NAMES[:3] for r in RATIOS)
PACKET = Path(__file__).resolve().parents[2] / "assets" / "bearing-calibration.json"


def calibrate(packet, slug, input_sha, vent, field="scores"):
    if packet is None:
        packet = json.loads(PACKET.read_text())
    if tuple(map(tuple, packet["candidates"])) != CANDIDATES or tuple(packet["controls"]) != NAMES:
        raise ValueError("Unexpected calibration structures")
    rows, weights, included = [], [], []
    for case in packet["cases"]:
        other = case["vent"]
        same_vent = abs((other[0] - vent[0] + 180) % 360 - 180) < .01 and abs(other[1] - vent[1]) < .01
        if case["slug"] == slug or case["input_sha256"] == input_sha or same_vent:
            continue
        groups = [[fold for fold in case["folds"] if fold["kind"] == kind and fold["passed"]]
                  for kind in ("geographic", "quadrants")]
        groups = [group for group in groups if group]
        if groups:
            included.append(case["slug"])
        for group in groups:
            for fold in group:
                values = np.asarray(fold[field])
                rows.extend(values)
                weights.extend(np.full(len(values), 1. / (len(groups) * len(group) * len(values))))
    count = len(CANDIDATES) if field == "scores" else len(NAMES)
    estimated = solve(np.asarray(rows), np.asarray(weights)) if rows else np.full(count, 1. / count)
    return estimated, included
