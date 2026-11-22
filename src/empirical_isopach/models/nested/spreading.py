"""Sublinear widening for each finite transport corridor of the nested deposit."""
import numpy as np

from ..continuation_geometry import half_thickness
from .geometry import positions
from .reach import WIDTHS, EXTENSIONS


def field_difference(model, layout, xy, strength=1.):
    """Keep the broad law, fixed width/length priors and training likelihoods."""
    radius, bearing = positions(xy, model.latitude)
    edge = np.minimum(radius, layout.lengths.max())
    old = model.corridors.penalties(edge, bearing)
    across, _ = layout.costs(xy, at_survey_edge=True)
    along = (edge[:, None]*np.column_stack((np.sin(bearing), np.cos(bearing)))) @ layout.directions.T
    # Retain the existing longitudinal thinning law beyond the last positive station.
    beyond = (np.maximum(along-layout.lengths, 0.)/(layout.proximal+layout.lengths))**2
    new = [np.zeros(len(xy))]
    for width in WIDTHS:
        transverse = across / np.tan(np.deg2rad(width))**2
        for extension in EXTENSIONS:
            new.append(half_thickness(np.min(transverse + beyond/extension**2, axis=1)))
    delta = np.minimum(0., np.array(new)-old)
    # Finite corridors carry a diffuse skirt. Law zero remains the whole-area control.
    aspect = layout.axial_leverage(xy)
    skirt = .25*(1.-.75*aspect)
    return strength*layout.freedom(xy)[None, :]*np.logaddexp(np.log(skirt), np.log1p(-skirt)+delta)


class SampledRidge:
    """Minimum intrinsic area rank reached on a sampled bearing at a given radius."""
    def __init__(self, model, layout):
        from .geometry import ranks
        self.model = model
        types, self.ids = np.unique(model.parent.geometry, return_inverse=True)
        self.radii = np.r_[0.,np.geomspace(max(1e-4,layout.proximal/100.),layout.lengths.max(),161)]
        directions = np.arctan2(layout.directions[:,0],layout.directions[:,1])
        floors = []
        for r in self.radii[1:]:
            intrinsic = ranks(np.full(len(directions),r),directions,types)
            overshoot = np.maximum(0.,r-layout.lengths)/(layout.proximal+layout.lengths)
            floors.append(np.min(intrinsic*(1+overshoot**2),axis=1)/r)
        self.ratio = np.vstack([floors[0],*floors]).T

    def flank(self, radius):
        ratios = np.array([np.interp(radius,self.radii,row) for row in self.ratio])
        return self.model.parent.log_at((ratios*radius)[self.ids])
