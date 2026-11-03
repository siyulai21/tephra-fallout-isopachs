"""Finite-domain integration, correlated surface sampling, and map artifacts."""
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.linalg import cholesky
from .distributions import predictive_distribution


def domain_bounds(xy, padding=.25):
    pts = np.vstack((xy, [0., 0.]))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi-lo
    span = np.maximum(span, max(span.max(), 1.)*.05)
    return np.r_[lo-padding*span, hi+padding*span]


def grid_cells(bounds, resolution):
    if resolution < 2:
        raise ValueError("Grid resolution must be >=2.")
    xmin, ymin, xmax, ymax = np.asarray(bounds, float)
    if not np.isfinite(bounds).all() or xmax <= xmin or ymax <= ymin:
        raise ValueError("Bounds must be finite xmin ymin xmax ymax with positive area.")
    dx, dy = (xmax-xmin)/resolution, (ymax-ymin)/resolution
    X, Y = np.meshgrid(xmin+(np.arange(resolution)+.5)*dx, ymin+(np.arange(resolution)+.5)*dy)
    return np.column_stack((X.ravel(), Y.ravel())), dx*dy


def support_flags(data, xy):
    try:
        inside = Delaunay(data.xy).find_simplex(xy) >= 0
    except QhullError:
        inside = np.zeros(len(xy), dtype=bool)
    distance = cKDTree(data.xy).query(xy)[0]
    return inside, distance


def integrate(model, bounds, resolution=24, draws=1000, seed=1729):
    """Volume of the latent thickness surface, in km³, only inside bounds.

    Joint Gaussian log-surface draws preserve spatial covariance. Never sum
    independent pointwise interval endpoints to claim a volume interval.
    """
    if draws < 20 or resolution > 60:
        raise ValueError("Use at least 20 draws and resolution <=60 for joint integration.")
    xy, area = grid_cells(bounds, resolution)
    if hasattr(model, "sample_log_fields"):
        log_samples = model.sample_log_fields(xy, draws, seed)
    else:
        m, C = model.predict_log(xy, covariance=True)
        numerical_jitter = max(model.amplitude**2, 1.)*1e-10
        L = cholesky(C + numerical_jitter*np.eye(len(C)), lower=True)
        rng = np.random.default_rng(seed)
        log_samples = m[:, None]+L@rng.standard_normal((len(m), draws))
    with np.errstate(over="raise", invalid="raise"):
        volumes = np.exp(log_samples).sum(axis=0)*area*1e-5
        expected_cells = np.exp(predictive_distribution(model, xy).log_expected_thickness())*area*1e-5
    inside, _ = support_flags(model.data, xy)
    return {"bounds_km": list(map(float, bounds)), "resolution": resolution,
            "draws": draws, "seed": seed, "mean_km3": float(expected_cells.sum()),
            "conditional_quantiles_km3": dict(zip(("0.025", "0.5", "0.975"),
                                                  np.quantile(volumes, [.025, .5, .975]).tolist())),
            "fraction_expected_volume_outside_data_hull": float(expected_cells[~inside].sum()/expected_cells.sum()),
            "scope": "Finite rectangle only; conditional on fitted component parameters and estimated weights. Mixture draws choose one component for the whole field; observation discrepancy is excluded." if hasattr(model, "models") else
                     "Finite rectangle only; conditional on fitted trend, covariance and selected family."}


def expected_volume(model, bounds, resolution=80):
    xy, area = grid_cells(bounds, resolution)
    with np.errstate(over="raise"):
        return float(np.exp(predictive_distribution(model, xy).log_expected_thickness()).sum()*area*1e-5)


def save_map(path, model, xy, shape, mean, variance, inside, alternate_mean=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, Y = xy[:, 0].reshape(shape), xy[:, 1].reshape(shape)
    panels = 3 if alternate_mean is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(6*panels, 5), layout="constrained")
    med = np.exp(mean).reshape(shape)
    im = axes[0].pcolormesh(X, Y, mean.reshape(shape)/np.log(10), shading="nearest", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="log₁₀ median thickness (cm)")
    levels = np.array([.1, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    levels = levels[(levels > med.min()) & (levels < med.max())]
    if len(levels):
        cs = axes[0].contour(X, Y, med, levels=levels, colors="white", linewidths=.7)
        axes[0].clabel(cs, fontsize=7, fmt="%g")
    axes[0].set_title(f"Median isopachs · {model.candidate.tail}")
    im = axes[1].pcolormesh(X, Y, np.sqrt(variance).reshape(shape), shading="nearest", cmap="magma")
    fig.colorbar(im, ax=axes[1], label="Conditional SD of ln thickness")
    axes[1].set_title("Uncertainty given fitted parameters")
    if alternate_mean is not None:
        difference = ((mean-alternate_mean)/np.log(10)).reshape(shape)
        limit = max(float(np.max(abs(difference))), .01)
        im = axes[2].pcolormesh(X, Y, difference, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=axes[2], label="log₁₀(selected median / alternative median)")
        axes[2].set_title("Sensitivity to thinning assumption")
    for ax in axes:
        if inside.any() and not inside.all():
            ax.contour(X, Y, inside.reshape(shape).astype(float), levels=[.5], colors="cyan", linestyles="--", linewidths=1)
        det = ~model.data.censored
        ax.scatter(*model.data.xy[det].T, c="white", edgecolors="black", s=14, linewidths=.6, label="Measurement")
        for mask, marker, label in ((model.data.censored & ~model.data.lower_censored, "v", "Upper limit"),
                                    (model.data.lower_censored, "^", "Lower bound")):
            if mask.any():
                ax.scatter(*model.data.xy[mask].T, c="orange", marker=marker, s=25, label=label)
        ax.scatter([0], [0], c="red", marker="*", s=100, edgecolors="white", label="Vent")
        ax.set(xlabel="East of vent (km)", ylabel="North of vent (km)", aspect="equal")
    axes[0].legend(fontsize=8)
    fig.suptitle("Real observations only · cyan boundary: data convex hull · contours may remain open", fontsize=13)
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)
