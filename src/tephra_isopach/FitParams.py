from dataclasses import dataclass


@dataclass
class FitParams:
    x_spacing: float = 10000
    y_spacing: float = 10000
    lambda_tension: float = 1e-1
    lambda_rough: float = 1

