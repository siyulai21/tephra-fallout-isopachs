"""Shared complete-contour screen for the mass-normalized spherical posterior."""
from pathlib import Path
from hashlib import sha256

import numpy as np

from ...contour_support import audit_contours
from ...core.distributions import EmpiricalMarginals
from ...core.geometry import survey_bounds
from .geometry import log_field, to_sphere


class PosteriorSurface:
    def __init__(self, draws, latitude):
        self.draws = draws
        self.latitude = latitude

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        if observation:
            raise ValueError("The complete-contour screen requires latent thickness")
        return EmpiricalMarginals(log_field(self.draws,to_sphere(xy,self.latitude)))


def checked_subset(path, expected):
    p = Path(path)
    if sha256(p.read_bytes()).hexdigest() != expected:
        raise ValueError("Spherical posterior checksum mismatch")
    with np.load(p) as archive:
        chains, n = archive["log_volume"].shape
        idx = np.unique(np.linspace(0,n-1,min(n,64)).round().astype(int))
        return {k:v[:,idx].reshape((chains*len(idx),*v.shape[2:])) for k,v in archive.items()}


def screen(data, latitude, report, groups):
    bounds = survey_bounds(data.xy)
    checks = [{"fold":f["fold"], "passed":f["fit"]["diagnostics"]["passed"],
               "posterior_sha256":f["fit"]["posterior_sha256"]} for f in report["validation"]["fits"]]
    metadata = {"protocol":"supported-isopachs-v1","input_sha256":report["input"]["input_sha256"],
                "baseline_posterior_sha256":report["fits"]["baseline"]["posterior_sha256"],
                "deletion_fit_status":checks,"fields_per_surface":len(groups["reference"]["log_volume"]),
                "map_bounds_km":bounds.tolist()}
    try:
        to_sphere(np.array([[x,y] for x in bounds[[0,2]] for y in bounds[[1,3]]]),latitude)
    except ValueError:
        return {**metadata,"status":"audit_window_leaves_physical_chart","policies":{},"refined_levels":[]}
    full = PosteriorSurface(groups["reference"],latitude)
    folds = [PosteriorSurface(groups[f"fold-{c['fold']}"],latitude) for c in checks if c["passed"]]
    result, *_ = audit_contours(full,data,bounds,folds)
    return {**result,**metadata,"status":"completed",
            "surface":"Pointwise posterior median from up to 64 evenly spaced draws per chain. Eligibility requires four numerically usable geographic deletions."}


def load_surfaces(report, baseline, folds):
    groups = {"reference":checked_subset(baseline,report["fits"]["baseline"]["posterior_sha256"])}
    for f in report["validation"]["fits"]:
        groups[f"fold-{f['fold']}"] = checked_subset(folds[f["fold"]],f["fit"]["posterior_sha256"])
    return groups
