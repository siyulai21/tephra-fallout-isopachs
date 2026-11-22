"""Four-deletion audit of the conditional median field under the common support protocol."""
import gzip
import json

import numpy as np

from ..contour_support import audit_contours
from ..core.geometry import survey_bounds
from .directional.azimuth.field import AzimuthField
from .directional.ensemble.fields import Ensemble, Member, Marginal, subset_draws
from .extent_integrals import parent_fields
from .extent_review import ExtentFields
from .extent_validation import fold_models
from .mass_mixture.bearings.field import BearingField, phases
from src.probabilistic_isopach.global_volume import EarthDomain


class SupportField:
    median_grid_batch = 160

    def __init__(self, family, parent, data, latitude, cal):
        if family=="directional":
            members = []
            for m in parent.members:
                draws,idx = subset_draws(m.draws,16)
                members.append(Member(m.geometry,draws,idx,m.records))
            parent = AzimuthField(Ensemble(members,parent.model_weights),data,parent.concentration_weights)
        elif family=="spherical-mixture":
            n = len(parent.phi)//4
            subset = np.round(np.linspace(0,n-1,min(n,64))).astype(int)
            index = (np.arange(4)[:,None]*n+subset).ravel()
            parent = BearingField({k:v[index] for k,v in parent.draws.items()},parent.support,phases((4,len(subset))))
        else:
            parent = parent.reduced(256)
        self.fields = ExtentFields(family,parent,data,latitude,cal["states"])
        self.weights = np.concatenate([w*self.fields.weights for w in cal["weights"]])

    def distribution(self, xy, observation=False, log_sd=None, calibrated=False):
        return Marginal(np.concatenate(self.fields.variants(xy)),self.weights)


def support_screen(folder, family, data, meta, work, cal):
    lat = meta["vent_lonlat"][1]
    full = parent_fields(folder,family,data,lat,states=cal["states"],count=64 if family=="spherical-mixture" else 256)
    bounds = survey_bounds(data.xy)
    domain = EarthDomain.wgs84(lat)
    corners = np.array([[x,y] for x in bounds[[0,2]] for y in bounds[[1,3]]])
    if domain.rho(corners).max()>=domain.rho_max:
        return dict(status="audit_window_leaves_physical_chart",policies={},refined_levels=[])
    field = SupportField(family,full.parent,data,lat,cal)
    models,checks = [],[]
    for _,i,tr,_,model,_,passed in fold_models(family,folder,data,lat,work,geographic_only=True):
        checks.append(dict(fold=i,passed=bool(model is not None and passed is True)))
        if model is not None and passed is True:
            models.append(SupportField(family,model,tr,lat,cal))
    audit,*_ = audit_contours(field,data,bounds,models)
    return {**audit,"status":"completed","deletion_checks":checks,"map_bounds_km":bounds.tolist(),
            "field_draws_per_continuation":len(field.fields.weights),"continuation":cal}


def save_screen(path, report):
    path.write_bytes(gzip.compress((json.dumps(report,allow_nan=False)+"\n").encode(),mtime=0))
