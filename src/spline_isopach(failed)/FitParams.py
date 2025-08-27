from dataclasses import dataclass


@dataclass
class FitParams:
    x_spacing: float  # unit in km
    y_spacing: float
    lambda_tension: float = 0.1
    lambda_rough: float = 1000
