"""Pointwise latent thickness exceedance in the same source-relative geographic windows."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from .contour_limits import local_half_width
from .deposit_maps import draw_land, frames, regular_axes
from .deposit_field import digest


def probability_plate(folder, field, model, data, meta, name, receipt):
    bounds, _, _ = frames(field, data, 0.5)
    local = local_half_width(data)
    middle = np.sqrt(local * max(abs(bounds)))
    windows = [np.array([-h, -h, h, h]) for h in (local, middle)] + [bounds]
    thresholds = [1.0, max(2.0, receipt["inner_contour"]["level_cm"] / 2.0)]
    limits = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    colors = ListedColormap(["#f6f7ed", "#d5e7dc", "#9bcabe", "#65aaa4", "#357b84", "#1a445b"])
    norm = BoundaryNorm(limits, colors.N)
    ranges = []
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 7.2), layout="constrained")
    for column, box in enumerate(windows):
        x, y = regular_axes(box, 177)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        within_earth = model.domain.rho(points) <= model.domain.rho_max
        valid = np.flatnonzero(within_earth)
        probs = np.full((2, len(points)), np.nan)
        for start in range(0, len(valid), 512):
            block = valid[start : start + 512]
            latent = model.distribution(points[block])
            for row, level in enumerate(thresholds):
                probs[row, block] = 1 - latent.cdf(np.log(level))
        ranges.append([[float(np.nanmin(z)), float(np.nanmax(z))] for z in probs])
        for row, level in enumerate(thresholds):
            ax = axes[row, column]
            draw_land(ax, box, meta["vent_lonlat"], 185)
            z = probs[row].reshape(xx.shape)
            ax.pcolormesh(x, y, z, cmap=colors, norm=norm, shading="nearest", alpha=0.70, rasterized=True)
            if np.nanmin(z) < 0.5 < np.nanmax(z):
                ax.contour(x, y, z, levels=[0.5], colors=["#152f40"], linewidths=1.1, linestyles="dashed")
            ax.scatter(*data.xy.T, s=6, c="#293541", lw=0, alpha=0.57)
            ax.plot(0, 0, "*", ms=8, mfc="#ffdd81", mec="#4b351f", mew=0.7)
            for inner in windows[:column]:
                ax.add_patch(
                    Rectangle(inner[:2], *(inner[2:] - inner[:2]), fill=False, ls=":", lw=0.85, ec="#684f7c")
                )
            ax.set(aspect="equal", xlim=box[::2], ylim=box[1::2])
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(("Proximal", "Intermediate", "Full deposit")[column], fontsize=11)
            if column == 0:
                ax.set_ylabel(f"Thickness ≥ {level:g} cm\nNorthing (km)", fontsize=9)
            if row == 1:
                ax.set_xlabel("Easting from vent (km)", fontsize=8)
    fig.suptitle(f"{name} · probability of exceeding a deposit thickness", fontsize=14)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=colors)
    cb = fig.colorbar(scalar, ax=axes, orientation="horizontal", shrink=0.64, aspect=45, pad=0.025)
    cb.set_ticks(limits, labels=[f"{v:.0%}" for v in limits])
    cb.set_label("Conditional probability for the latent deposit", fontsize=9)
    fig.legend(
        handles=[
            Line2D([], [], ls="", marker=".", c="#293541", label="Fitted locality"),
            Line2D([], [], ls="", marker="*", mfc="#ffdd81", mec="#4b351f", label="Vent"),
            Line2D([], [], ls="--", c="#152f40", label="50% probability"),
            Line2D([], [], ls=":", c="#684f7c", label="Window enlarged to the left"),
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    info = dict(
        input_sha256=meta["input_sha256"],
        posterior_sha256=receipt["sources_sha256"]["multiscale/posterior.npz"],
        median_grid_sha256=receipt["median_grid_sha256"],
        thresholds_cm=thresholds,
        windows_km=[b.tolist() for b in windows],
        probability_bounds_by_window=ranges,
        accepted_rows=len(data.xy),
        probabilities=limits.tolist(),
        quantity="latent thickness exceedance",
    )
    root = Path(folder)
    path = root / "exceedance.png"
    fig.savefig(path, dpi=140, metadata={"Description": json.dumps(info, sort_keys=True)})
    plt.close(fig)
    info["figure_sha256"] = digest(path)
    (root / "exceedance.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
