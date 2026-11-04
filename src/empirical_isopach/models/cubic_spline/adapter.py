"""Automatic cubic-spline fitting and model-specific numerical diagnostics."""
import numpy as np

from .model import TrendRegularizedSpline
from .selection import select_spline


def fit(data, bounds, folds, buffer_km):
    return select_spline(data, bounds=bounds, n_folds=folds, buffer_km=buffer_km)


def diagnose(model, data, bounds, region, result):
    from ...region import integrate_region
    refined = TrendRegularizedSpline(model.candidate, quadrature_order=12).fit(data, bounds=bounds)
    qxy, _ = region.quadrature(16)
    prediction_change = float(np.max(abs(model.predict_log(qxy)[0]-refined.predict_log(qxy)[0])))
    fine_volume = integrate_region(refined, region)
    relative_volume_change = abs(fine_volume["expected_km3"]/result["regional_volume"]["expected_km3"]-1)
    check = {"orders": [6, 12], "max_log_prediction_change": prediction_change,
        "relative_expected_volume_change": relative_volume_change,
        "passed": prediction_change < .02 and relative_volume_change < .005}
    notes = []
    if not check["passed"]:
        notes.append("Spline trend-penalty quadrature sensitivity exceeds the declared tolerance; inspect its numerical check.")
    edges = result["selection"]["selected_at_explored_boundary"]
    if edges:
        notes.append("Spline selection reaches an explored search boundary: "+", ".join(edges)+". This is a bounded-grid result, not proof of a continuous optimum.")
    if model.candidate.trend_strength >= 10000:
        notes.append("The selected spline strongly follows its estimated trend. Its conditional volume interval can be extremely narrow because trend-estimation uncertainty is omitted; inspect the geographic-deletion range rather than interpreting narrowness as established precision.")
    return {"penalty_quadrature_check": check}, notes
