"""Universal, explicit assumptions; no eruption-specific wind parameters."""
from dataclasses import dataclass, asdict, replace
import numpy as np


@dataclass(frozen=True)
class Prior:
    name: str = "baseline"
    components: int = 3
    log_volume_mean: float = 0.
    log_volume_sd: float = 6.
    log_scale_mean: float = float(np.log(300.))
    log_scale_sd: float = 1.5
    offset_sd: float = 3.
    tail_log_mean: float = 0.
    tail_log_sd: float = 1.
    integrable_plane_tail: bool = True
    contamination: bool = True
    exponential_core: bool = True
    fan_concentration: float = .3
    kernel_concentration: float = .3
    noise_scale: float = 1.5
    source_anchored: bool = False
    angular_sd: float = .75
    angular_nodes: int = 256
    source_shape: str = "harmonic"
    vent_envelope: bool = False
    supported_basis: bool = False
    basis_width_log_mean: float = float(np.log(.5))
    basis_width_log_sd: float = .7

    def __post_init__(self):
        if not isinstance(self.components, int) or self.components < 1:
            raise ValueError("Fan count must be a positive integer.")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.log_volume_sd, self.log_scale_sd, self.offset_sd, self.tail_log_sd, self.noise_scale,
                self.fan_concentration, self.kernel_concentration, self.angular_sd, self.basis_width_log_sd)):
            raise ValueError("Prior scales must be positive and finite.")
        if self.angular_nodes < 32 or self.angular_nodes % 2:
            raise ValueError("Use an even angular quadrature size of at least 32.")
        if self.source_shape not in ("harmonic", "elliptic"):
            raise ValueError("Unknown source fan shape.")
        if self.vent_envelope and self.source_anchored:
            raise ValueError("Choose the vent envelope or monotone directional fans, not both.")
        if self.supported_basis and not self.vent_envelope:
            raise ValueError("Supported bases require the vent envelope.")
        if not np.isfinite([self.log_volume_mean, self.log_scale_mean, self.tail_log_mean, self.basis_width_log_mean]).all():
            raise ValueError("Prior locations must be finite.")

    def describe(self):
        result = {**asdict(self), "weights": "Symmetric Dirichlet(total concentration fan_concentration)",
            "kernel_weights": "Symmetric Dirichlet(total concentration kernel_concentration) over Gaussian, exponential, power components, shared across fans" if self.exponential_core else "Symmetric Dirichlet over Gaussian and power components",
            "tail_exponent": "1+exp(Normal(tail_log_mean,tail_log_sd))" if self.integrable_plane_tail else "exp(Normal(tail_log_mean,tail_log_sd))",
            "noise": "HalfNormal(noise_scale); broad-error fraction Beta(1,9), fourfold residual SD" if self.contamination else "HalfNormal(noise_scale); Gaussian log-observation error",
            "interpretation": "Proper modelling assumptions, not extra observations or geological facts. Identical prior specification for every eruption."}
        if self.vent_envelope:
            result.update(kernel_weights="Fixed Gaussian profiles; no fitted kernel mixture",
                tail_exponent="Unused: the bounded envelope uses Gaussian profiles",
                weights="Dirichlet peak-amplitude weights, not component mass fractions",
                envelope_strength="Beta(1,1)",
                core_scale="exp(Normal(log_scale_mean,log_scale_sd)) km",
                secondary_geometry=("Training-location bases; shared exp(Normal(basis_width_log_mean,basis_width_log_sd)) width multiplier"
                    if self.supported_basis else "Freely displaced circular Gaussian profiles"))
        elif self.source_anchored:
            result["source_geometry"] = "Source-focused ellipses" if self.source_shape == "elliptic" else "Source-anchored harmonic fans"
            result["angular_prior"] = "Independent Normal(0,angular_sd) coefficients; no fitted displacement"
        return result


BASELINE = Prior()
SENSITIVITIES = (
    replace(BASELINE, name="diffuse_tail", integrable_plane_tail=False, tail_log_mean=float(np.log(2.))),
    replace(BASELINE, name="wide_geography", log_scale_sd=2.25, offset_sd=6.),
    replace(BASELINE, name="volume_prior_shift", log_volume_mean=float(np.log(100.))),
    replace(BASELINE, name="one_fan", components=1),
    replace(BASELINE, name="five_fans", components=5),
    replace(BASELINE, name="gaussian_errors", contamination=False),
    replace(BASELINE, name="diffuse_mixture_weights", fan_concentration=3., kernel_concentration=3.),
)
