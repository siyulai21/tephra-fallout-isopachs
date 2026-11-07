"""Fixed source-focused candidate; shared choices, no eruption-specific tuning.

The dataclass specifications are fully serialized in every cache signature.
The preceding displaced-circle priors remain available in prior.py.
"""
from dataclasses import replace
from .prior import BASELINE as DISPLACED_BASELINE, SENSITIVITIES as DISPLACED_SENSITIVITIES

BASELINE = replace(DISPLACED_BASELINE, source_anchored=True, source_shape="elliptic", angular_sd=1.25)
SENSITIVITIES = tuple(replace(p, source_anchored=True, source_shape="elliptic",
                            angular_sd=2. if p.name == "wide_geography" else 1.25)
                      for p in DISPLACED_SENSITIVITIES)


def protocol(geometry):
    """Named, shared scientific specifications; never tune to a volume target."""
    if geometry == "displaced":
        return DISPLACED_BASELINE, DISPLACED_SENSITIVITIES
    if geometry == "source-fans":
        return BASELINE, SENSITIVITIES
    if geometry == "source-envelope":
        base = replace(DISPLACED_BASELINE, components=12, vent_envelope=True,
                       supported_basis=True, fan_concentration=1.)
        return base, (
            replace(base, name="six_bases", components=6),
            replace(base, name="three_bases", components=3),
            replace(base, name="wide_geography", basis_width_log_sd=1.4, log_scale_sd=2.25),
            replace(base, name="volume_prior_shift", log_volume_mean=4.605170185988092),
            replace(base, name="gaussian_errors", contamination=False),
            replace(base, name="diffuse_mixture_weights", fan_concentration=6.),
            replace(BASELINE, name="source_heavy_tails"),
        )
    raise ValueError(f"Unknown global geometry: {geometry}")
