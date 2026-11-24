"""Common directional structures and matched, buffered training partitions."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from ....core.validation import spatial_folds, training_mask
from ....validation import automatic_buffer
from ..geometry import BASE, POLICIES, Geometry, polar
from ..study import SEED, fit, write_json


MEMBERS = (POLICIES[5], BASE, POLICIES[6],
           replace(POLICIES[5], name="narrow_missing_fan", hidden_fan=True),
           POLICIES[7], replace(POLICIES[6], name="broad_missing_fan", hidden_fan=True))
NAMES = tuple(p.name for p in MEMBERS)
# Preserve the seeds of the original four identical fits.
CODES = (5, 0, 6, 13, 7, 14)


def splits(data, latitude, kind):
    buffer, _ = automatic_buffer(data)
    if kind == "geographic":
        labels = spatial_folds(data.xy, 4)
    elif kind == "quadrants":
        _, angle = polar(data.xy, Geometry(data, latitude).axis)
        labels = np.floor(angle/(np.pi/2)).astype(int).clip(0, 3)
    else:
        raise ValueError("Unknown deletion design")
    for i in range(4):
        test = labels == i
        if test.any():
            train, test = training_mask(data, labels, i, buffer)
            yield i, data.subset(train), data.subset(test)
        else:
            yield i, None, data.subset(test)


def directory(work, member, kind="full", fold=None):
    return (Path(work)/"fits"/member.name if kind == "full" else
            Path(work)/kind/f"fold-{fold}"/member.name)


def run_members(data, latitude, work, kinds=("full", "geographic", "quadrants"), indices=range(6)):
    data = data.canonical()
    for kind in kinds:
        partitions = [(None, data, None)] if kind == "full" else splits(data, latitude, kind)
        for i, train, test in partitions:
            for m in indices:
                p, code = MEMBERS[m], CODES[m]
                folder = directory(work, p, kind, i)
                if train is None:
                    write_json(folder/"status.json", {"status": "empty"})
                    continue
                try:
                    g = Geometry(train, latitude, p)
                except ValueError as exc:
                    if kind == "full":
                        raise
                    write_json(folder/"status.json", {"status": "insufficient support", "reason": str(exc)})
                    continue
                offset = code if kind == "full" else code*1000+(101 if kind == "geographic" else 201)+i
                _, info = fit(train, g, SEED+offset, folder)
                write_json(folder/"status.json", {"status": "estimated", **info})
