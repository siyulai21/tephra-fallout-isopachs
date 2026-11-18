"""Apply the repository's complete-contour screen to the directional posterior."""
from pathlib import Path
import json

import numpy as np
from ...core.distributions import EmpiricalMarginals

from ...contour_support import audit_contours
from ...core.geometry import survey_bounds
from .geometry import Geometry, RADIUS, polar
from .inference import field
from .study import digest


class ScreenField:
    def __init__(self, geometry, folder):
        self.geometry = geometry
        record = json.loads(sorted(Path(folder).glob("attempt-*.json"))[-1].read_text())
        path = Path(folder)/record["file"]
        if digest(path) != record["posterior_sha256"]:
            raise ValueError("Changed posterior for contour audit")
        with np.load(path) as archive:
            idx = np.linspace(0,record["draws_per_chain"]-1,64).round().astype(int)
            self.fields = {k:v[:,idx].reshape((record["chains"]*len(idx), *v.shape[2:]))
                           for k,v in archive.items()}

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        if observation:
            raise ValueError("The contour audit requires latent thickness")
        return EmpiricalMarginals(field(self.fields,self.geometry,xy))


def shared_audit(data, latitude, work, validation):
    g = Geometry(data,latitude)
    bounds = survey_bounds(data.xy)
    if np.max(polar(np.array([[x,y] for x in bounds[[0,2]] for y in bounds[[1,3]]]),g.axis)[0]) > 2*RADIUS:
        return {"protocol":"supported-isopachs-v1", "status":"audit_window_leaves_physical_chart",
                "policies":{}, "refined_levels":[]}
    base = ScreenField(g, work/"fits/reference")
    models = []
    checks = [{k: f[k] for k in ("fold", "status", "passed") if k in f} for f in validation["fits"]]
    for f in validation["fits"]:
        if f["status"] != "estimated" or not f["passed"]:
            continue
        train = data.subset(np.isin(data.row_ids,f["training_row_ids"]))
        models.append(ScreenField(Geometry(train,latitude),work/"geographic"/f"fold-{f['fold']}"))
    result, *_ = audit_contours(base, data, bounds, models)
    result.update(status="completed", posterior_fields_per_surface=256,
                  map_bounds_km=bounds.tolist(), deletion_fit_status=checks,
                  surface="Pointwise posterior median of 64 deterministically spaced draws per independent chain. Eligibility requires four usable buffered geographic refits.")
    return result
