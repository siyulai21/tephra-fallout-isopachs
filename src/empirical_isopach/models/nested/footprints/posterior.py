"""Principal continuous-growth inference and readers for earlier footprint packets.

The old field reader remains available for immutable comparisons. All new fits
use the continuous-direction posterior; the historical engine is isolated in
legacy_posterior.py, with its original source digest unchanged.
"""
from dataclasses import dataclass
import numpy as np

from ..continuation import Continued
from ..geometry import positions
from . import geometry, growth
from .continuous import FlowField, PROTOCOL, engine, fit_field, write_fit, predictions

LEGACY_PROTOCOL = "nested-dispersal-footprints-v2"
GROWTH_PROTOCOL = "nested-area-growth-v3"


@dataclass
class FootprintField(Continued):
    geography_version: int = 2

    @property
    def geography(self):
        return growth if self.geography_version == 3 else geometry

    def field(self, xy):
        geometry_ids, inverse = np.unique(self.geometry, return_inverse=True)
        rank = self.geography.ranks(*positions(xy, self.latitude), geometry_ids)
        return self.log_at(rank[inverse])

    def subset(self, idx):
        return FootprintField(**vars(super().subset(idx)), geography_version=self.geography_version)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as packed:
            if "warp" in packed:
                return FlowField.load(path)
            return cls(**{k: (float(packed[k]) if k == "latitude" else int(packed[k]) if k == "geography_version" else packed[k]) for k in packed})
