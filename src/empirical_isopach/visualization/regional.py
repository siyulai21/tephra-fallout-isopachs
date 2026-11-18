"""Complete-hull maps for regional peers, with common scales and frames."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from ..core.distributions import predictive_distribution
from src.probabilistic_isopach.figures import STYLE, CMAP
from .maps import (ASSET, INK, AMBER, frame_bounds, grid_axes, panel_layout,
                   setup_map, observations, observation_legend, contour_levels, level_labels)


def regional_grid(fit_bounds, region, models):
    """Evaluate the published regional display grid in the original fitting domain."""
    # Plot frame and spline fitting domain are distinct. Never enlarge or refit
    # a spline merely to include a source outside the observation hull.
    x, y = grid_axes(fit_bounds, longest=361, minimum=121)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    inside = region.contains(xy)
    fields, table = {}, {"x_km":xy[:, 0], "y_km":xy[:, 1], "inside_observation_hull":inside}
    for method, model in models.items():
        parts = []
        for start in range(0, len(xy), 2048):
            d = predictive_distribution(model, xy[start:start+2048])
            parts.append(np.array([d.quantile(.5), d.log_expected_thickness(), d.quantile(.025), d.quantile(.975)]))
        values = np.concatenate(parts, axis=1)
        fields[method] = values[0].reshape(xx.shape)/np.log(10.)
        for key, row in zip(("median", "conditional_mean", "conditional_lower95", "conditional_upper95"), values):
            table[f"{method}_{key}_cm"] = np.exp(row)
    return x, y, inside, fields, table


def regional_maps(output, name, data, metadata, region, models, report):
    from ..reporting import report_label
    output = Path(output)
    fit_bounds = np.array(report["fit_bounds_km"])
    bounds = frame_bounds(np.vstack((region.vertices, data.xy)))
    x, y, inside, fields, table = regional_grid(fit_bounds, region, models)
    pd.DataFrame(table).to_csv(output/"predictions.csv", index=False)
    values = np.concatenate([v.ravel()[inside] for v in fields.values()])
    low, high = float(values.min()), max(float(values.max()), float(values.min())+.1)
    norm = Normalize(low, high)
    methods = tuple(report["methods"])
    shape, size = panel_layout(bounds, len(methods))
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(*shape, figsize=size, layout="constrained", squeeze=False)
        for ax, method in zip(axs.flat, methods):
            image = ax.pcolormesh(x, y, fields[method], shading="auto", cmap=CMAP,
                                  norm=norm, rasterized=True, zorder=1)
            # Clip the raster at the exact polygon, not at cell centres. The
            # complete hull therefore stays filled up to its extreme vertices.
            image.set_clip_path(Polygon(region.vertices, closed=True, transform=ax.transData))
            z = np.ma.array(fields[method], mask=(~inside).reshape(len(y), len(x)))
            levels = contour_levels(z.filled(np.nan))
            if len(levels):
                lines = ax.contour(x, y, z, levels=levels, colors=INK, linewidths=.65, zorder=4)
                text = ax.clabel(lines, fmt=level_labels, fontsize=8)
                for label in text:
                    label.set_path_effects([pe.withStroke(linewidth=2.3, foreground="white", alpha=.8)])
            boundary = np.vstack((region.vertices, region.vertices[0]))
            ax.plot(*boundary.T, color=AMBER, lw=1.25, ls=(0, (5, 3)), zorder=5)
            setup_map(ax, bounds, metadata["vent_lonlat"])
            observations(ax, data)
            volume = report["methods"][method]["regional_volume"]["expected_km3"]
            ax.set_title(f"{report_label(report, method)}\nExpected hull volume: {volume:,.1f} km³", fontsize=11)
        bar = fig.colorbar(image, ax=list(axs.flat), location="bottom", shrink=.85, pad=.055,
                          aspect=48, label="Posterior median thickness (cm; identical scale for every regional family)")
        levels = contour_levels(values)
        if len(levels) > 10:
            levels = levels[::2]
        bar.set_ticks(levels, labels=[level_labels(t) for t in levels])
        observation_legend(fig, data, [Line2D([], [], c=AMBER, ls="--", label="Complete observation hull")])
        fig.suptitle(f"{name} · regional isopachs\nAll accepted locations and the complete hull are shown; gaps inside remain model-dependent.", fontsize=14)
        for suffix in ("png", "svg"):
            fig.savefig(output/f"isopachs.{suffix}", dpi=190, bbox_inches="tight", pad_inches=.16)
        plt.close(fig)
    contained = bool(np.all(region.vertices >= bounds[:2]) and np.all(region.vertices <= bounds[2:]))
    if not contained:
        raise AssertionError("The map must contain every hull vertex")
    display = {"protocol":"report-cartography-v1", "input_sha256":metadata["input_sha256"],
        "display_bounds_km":bounds.tolist(), "unchanged_fit_bounds_km":fit_bounds.tolist(),
        "entire_observation_hull_in_frame":contained, "hull_raster_clip":"Exact polygon, not grid-centre mask",
        "all_accepted_locations_in_frame":bool(np.all(data.xy >= bounds[:2]) and np.all(data.xy <= bounds[2:])),
        "grid_shape":[len(y), len(x)], "coastline_scale":"1:10 million", "coastline_sha256":sha256(ASSET.read_bytes()).hexdigest(),
        "central_statistic":"Posterior median thickness", "volume_calculation_changed":False}
    (output/"map-display.json").write_text(json.dumps(display, indent=2)+"\n")
    return display
