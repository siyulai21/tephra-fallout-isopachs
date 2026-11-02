"""Convergence-checked integrals of the unchanged conditional spatial model."""
import numpy as np

from .distributions import thickness_summaries
from .report import grid_cells, support_flags


def converged_volume(model, bounds, tolerance=.005, resolutions=(64, 128, 256, 512)):
    """Refine midpoint quadrature for both the mean and median surfaces.

    A median-surface integral is not the median of random total volume.
    All results are bulk deposit volume, not dense-rock equivalent (DRE).
    """
    if tolerance <= 0 or len(resolutions) < 2:
        raise ValueError("Use a positive tolerance and at least two refinement grids.")
    history = []
    for resolution in resolutions:
        xy, area = grid_cells(bounds, resolution)
        chunks = [thickness_summaries(model, xy[start:start+4096]) for start in range(0, len(xy), 4096)]
        m, log_expected, _ = (np.concatenate([p[i] for p in chunks]) for i in range(3))
        with np.errstate(over="raise", invalid="raise"):
            expected = np.exp(log_expected)*area*1e-5
            median = np.exp(m)*area*1e-5
        inside, _ = support_flags(model.data, xy)
        row = {"resolution": resolution, "mean_km3": float(expected.sum()),
               "median_surface_km3": float(median.sum()),
               "fraction_expected_volume_outside_data_hull": float(expected[~inside].sum()/expected.sum())}
        if history:
            row["relative_change"] = max(abs(row[k]-history[-1][k])/max(row[k], 1e-30)
                                         for k in ("mean_km3", "median_surface_km3"))
        history.append(row)
        if len(history) >= 2 and row["relative_change"] < tolerance:
            break
    return {**history[-1], "bounds_km": list(map(float, bounds)), "history": history,
            "converged": history[-1]["relative_change"] < tolerance,
            "tolerance": tolerance,
            "scope": "Bulk deposit volume inside the stated equal-area rectangle; not total erupted volume or DRE."}


def volume_analysis(model, alternate, bounds, sampled_volume):
    regional = converged_volume(model, bounds)
    center, half = (np.asarray(bounds[:2])+bounds[2:])/2, (np.asarray(bounds[2:])-bounds[:2])/2
    sensitivity = []
    for factor in (1., 1.5, 2.):
        expanded = np.r_[center-factor*half, center+factor*half]
        selected = regional if factor == 1 else converged_volume(model, expanded)
        alternative = converged_volume(alternate, expanded) if alternate else None
        sensitivity.append({"domain_factor": factor, "selected": selected, "alternative": alternative})
    sample_error = abs(sampled_volume["mean_km3"]-regional["mean_km3"])/regional["mean_km3"]
    return {"regional": regional, "domain_sensitivity": sensitivity,
            "conditional_sampling": {"quantiles_km3": sampled_volume["conditional_quantiles_km3"],
                "resolution": sampled_volume["resolution"], "draws": sampled_volume["draws"],
                "mean_grid_relative_error": sample_error,
                "mean_grid_agrees_with_refined_integral": sample_error < .01,
                "quantile_grid_convergence_verified": False,
                "interpretation": sampled_volume["scope"]+" Fitted-parameter uncertainty is not integrated. Mean-grid agreement does not establish quantile convergence."},
            "total_erupted_volume_km3": None,
            "total_status": "Not established: unsampled tails and eruption-wide uncertainty remain unresolved."}
