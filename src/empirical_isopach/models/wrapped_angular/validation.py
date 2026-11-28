"""Buffered outer folds and common contour-screen inputs with recorded row membership."""

import csv
import numpy as np
from pyproj import Geod
from ...core.validation import training_mask
from ...validation import automatic_buffer
from ..directional.ensemble.members import splits
from ...core.distributions import EmpiricalMarginals

# Duplicate descriptions and named sub-sections of the same field occurrence in
# the two-phase Millennium audit. The four reference families use the same groups.
LINKED = (
    ("Lv53-23", "Lv53-27", "old-058"),
    ("miy-guan", "miy-west", "miy-E", "old-002"),
    ("sun-HSG", "costa-P78"),
    ("sun-YC", "costa-P79"),
    ("machida-11", "machida-12"),
)


def partitions(data, meta, path, design="geographic"):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = dict(enumerate(csv.DictReader(stream), 2))
    linked = []
    ids = {r.get("EvidenceID"): i for i, r in rows.items() if r.get("EvidenceID")}
    for group in LINKED:
        if all(k in ids for k in group):
            linked.append([ids[k] for k in group])
    buffer, _ = automatic_buffer(data)
    if design == "distal":
        lon, lat = meta["vent_lonlat"]
        xy = np.array([[float(rows[i]["Lon"]), float(rows[i]["Lat"])] for i in data.row_ids])
        radius = Geod(ellps="WGS84").inv(np.full(len(xy), lon), np.full(len(xy), lat), *xy.T)[2]
        tr, te = training_mask(data, radius >= np.quantile(radius, 0.75), True, buffer)
        blocks = [(0, data.subset(tr), data.subset(te))]
    else:
        blocks = list(splits(data, meta["vent_lonlat"][1], design))
        if design == "geographic" and linked:
            which = {row: k for k, _, test in blocks for row in test.row_ids}
            initial = np.array([which[row] for row in data.row_ids])
            labels = initial.copy()
            for group in linked:
                mask = np.isin(data.row_ids, group)
                choices = np.unique(initial[mask])
                if len(choices) > 1:
                    centre = data.xy[mask].mean(0)
                    distances = [
                        np.linalg.norm(data.xy[(initial == k) & ~mask].mean(0) - centre) for k in choices
                    ]
                    labels[mask] = choices[np.argmin(distances)]
            blocks = []
            for k in range(4):
                tr, te = training_mask(data, labels, k, buffer)
                blocks.append((k, data.subset(tr), data.subset(te)))
    for k, train, test in blocks:
        if train is not None:
            for group in linked:
                if np.isin(test.row_ids, group).any():
                    train = train.subset(~np.isin(train.row_ids, group))
        yield k, train, test


class ContourView:
    """Fixed coherent quadrature for the shared full/deletion support screen."""

    def __init__(self, deposit):
        self.deposit = deposit
        self.bank = deposit.regional.coherent(128, 49037)
        self.median_grid_batch = 384

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        if observation:
            raise ValueError("Contour support requires the latent field")
        z = self.bank(xy).reshape(-1, len(xy))
        inner = self.deposit.local(xy)
        if inner.any():
            # Replace the sample median with the source-conditioned quantile at
            # those sites. The support screen and regional integral use its median.
            q = self.deposit.quantiles(xy[inner], (0.5,))[0]
            z[:, inner] += q - np.median(z[:, inner], axis=0)
        return EmpiricalMarginals(z)
