"""Partition the existing verified spherical-mixture posterior by compass bearing."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.mass_mixture.geometry import centers, kernel_samples
from src.empirical_isopach.models.mass_mixture.inference import flatten
from src.empirical_isopach.models.mass_mixture.source_geometry import mass_samples
from src.empirical_isopach.models.directional.geometry import COMPASS
from src.empirical_isopach.models.directional.estimate import interval
from src.empirical_isopach.models.directional.study import digest, write_json


def partition(folder):
    path = folder/"posterior.npz"
    report = json.loads((folder/"report.json").read_text())
    if digest(path) != report["published_artifact_sha256"]["posterior.npz"]:
        raise ValueError("Published spherical-mixture posterior mismatch")
    with np.load(path) as data:
        p = flatten(dict(data))
    idx = np.linspace(0, len(p["log_volume"])-1, 768).round().astype(int)
    uniform = qmc.Sobol(3, scramble=True, seed=20261117).random_base2(12)
    fine, coarse = np.zeros((len(idx), 8)), np.zeros((len(idx), 8))
    for j, i in enumerate(idx):
        for k, w in enumerate(p["weights"][i]):
            ls, q = p["log_scale"][i, k], p["exponent"][i]
            if "angular" in p:
                points = mass_samples(ls, p["angular"][i,k], q, p["kernel_weights"][i], uniform)
            else:
                center = centers(ls, p["offset"][i,k])
                points = kernel_samples(ls, center, q, p["tail_fraction"][i], uniform,
                                        p["exponential_fraction"][i])
            angle = np.arctan2(points[:,0],points[:,1])
            sector = np.floor((angle+np.pi/8)/(np.pi/4)).astype(int) % 8
            fine[j] += w*np.bincount(sector, minlength=8)/len(sector)
            half = len(sector)//2
            coarse[j] += w*np.bincount(sector[:half], minlength=8)/half
    if np.max(np.abs(fine.sum(1)-1)) > 1e-10:
        raise ArithmeticError("Mixture partition does not close")
    vol = np.exp(p["log_volume"][idx])
    return {"input_sha256":report["input"]["input_sha256"], "posterior_sha256":digest(path),
            "draws":len(idx), "sobol_points_per_component":len(uniform),
            "largest_sector_fraction_change_on_doubling":float(abs(fine-coarse).max()),
            "volume_km3":{s:interval(fine[:,j]*vol) for j,s in enumerate(COMPASS)},
            "fraction":{s:interval(fine[:,j]) for j,s in enumerate(COMPASS)}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    write_json(args.output, {r["slug"]:partition(Path("eruptions")/r["slug"]/"results/spherical-mixture")
                             for r in datasets()})
