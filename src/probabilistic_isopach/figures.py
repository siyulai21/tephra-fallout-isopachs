"""Offline, publication-quality maps and numerical diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.Laea import make_laea_transform
from .report import support_flags


INK = "#183843"
MUTED = "#5C7075"
TEAL = "#087E8B"
AMBER = "#AF6C25"
PAPER = "#FAFBF8"
CMAP = LinearSegmentedColormap.from_list("tephra", ["#E7F2EA", "#B0D9CF", "#5EB3AE", "#257F8E", "#183B57"])
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
         "axes.labelcolor": MUTED, "axes.edgecolor": "#B8C8CA", "xtick.color": MUTED,
         "ytick.color": MUTED, "axes.titleweight": "bold", "svg.fonttype": "none",
         "savefig.facecolor": PAPER, "figure.facecolor": PAPER, "axes.facecolor": PAPER}


def _coastlines(ax, vent):
    source = Path(__file__).parent/"assets/ne_110m_land.geojson"
    if not source.exists():
        return
    fwd, _, _ = make_laea_transform(*vent)
    segments = []
    for feature in json.loads(source.read_text())["features"]:
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            for ring in polygon:
                p = np.asarray(ring)
                xy = np.column_stack(fwd.transform(*p.T))
                pairs = np.stack((xy[:-1], xy[1:]), axis=1)
                valid = np.isfinite(pairs).all(axis=(1, 2)) & (np.linalg.norm(np.diff(pairs, axis=1)[:, 0], axis=1) < 900)
                segments.extend(pairs[valid])
    coast = LineCollection(segments, colors="#A0ADAB", linewidths=.55, zorder=2)
    coast.set_gid("coastlines")
    ax.add_collection(coast)


def _draw_ring(ax, ring, model, color, width=1.):
    inside, _ = support_flags(model.data, (ring[:-1]+ring[1:])/2)
    changes = np.r_[0, np.flatnonzero(inside[1:] != inside[:-1])+1, len(inside)]
    for a, b in zip(changes[:-1], changes[1:]):
        ax.plot(*ring[a:b+1].T, color=color, lw=width,
                ls="-" if inside[a] else (0, (4, 2.5)), zorder=4)


def save_isopach_map(output, name, model, metadata, contours, volume):
    output = Path(output)
    closed = [r for r in contours["levels"] if r["status"] == "closed"]
    levels = np.array([r["level_cm"] for r in closed])
    if not len(levels):
        raise ValueError("No resolved closed contours to plot.")
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 8.8))
        ax = fig.add_axes((.085, .18, .69, .65))
        sidebar = fig.add_axes((.81, .18, .18, .65))
        sidebar.set_axis_off()
        sidebar.set(xlim=(0, 1), ylim=(0, 1))
        fig.text(.075, .944, "TEPHRA  /  DEPOSIT ATLAS", fontsize=10, weight="bold", color=TEAL)
        fig.text(.075, .89, name, fontsize=29, weight="bold")
        fig.text(.075, .855, f"{metadata['n_measurements']} measurements  ·  {model.candidate.tail} thinning  ·  median surface",
                 fontsize=11, color=MUTED)
        upper = max(float(contours["mean"].max()), np.log(levels[-1])+.01)
        fill_levels = np.r_[np.log(levels), upper+.001]
        ax.contourf(contours["x"], contours["y"], contours["mean"], levels=fill_levels,
                    cmap=CMAP, alpha=.7, zorder=1)
        _coastlines(ax, metadata["vent_lonlat"])
        norm = Normalize(np.log(levels[0]), max(np.log(levels[-1]), np.log(levels[0])+1.))
        for level in levels:
            color = CMAP(.3+.7*norm(np.log(level)))
            for ring in contours["rings"][float(level)]:
                _draw_ring(ax, ring, model, color)
        cs = ax.contour(contours["x"], contours["y"], contours["mean"],
                        levels=np.log(levels), colors="none", linewidths=.1, linestyles="solid")
        labels = ax.clabel(cs, fmt={np.log(t): f"{t:g}" for t in levels}, colors=INK, fontsize=8,
                          inline=False)
        for label in labels:
            label.set_path_effects([pe.withStroke(linewidth=3, foreground=PAPER, alpha=.9)])
        try:
            hull = ConvexHull(model.data.xy)
            points = model.data.xy[np.r_[hull.vertices, hull.vertices[0]]]
            ax.plot(*points.T, color=TEAL, lw=.9, alpha=.8, ls=(0, (1, 3)), zorder=5)
        except QhullError:
            pass
        bx0, by0, bx1, by1 = volume["regional"]["bounds_km"]
        ax.add_patch(Rectangle((bx0, by0), bx1-bx0, by1-by0, fill=False, edgecolor=AMBER,
                               lw=.8, linestyle=(0, (6, 4)), zorder=3))
        detected = ~model.data.censored
        ax.scatter(*model.data.xy[detected].T, s=20, facecolors="white", edgecolors=INK,
                   linewidths=.7, zorder=7)
        if model.data.censored.any():
            ax.scatter(*model.data.xy[~detected].T, marker="v", s=24, c=AMBER, zorder=7)
        ax.scatter([0], [0], marker="*", s=155, c="#B9573C", edgecolors="white", linewidths=.8, zorder=8)
        bounds = contours["bounds_km"]
        ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal",
               xlabel="Projected easting from source reference (km)", ylabel="Projected northing from source reference (km)")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.18, linewidth=.5)
        sidebar.text(0, .98, "THICKNESS", weight="bold", fontsize=10)
        sidebar.text(0, .935, "Closed isopachs · cm", color=MUTED, fontsize=9)
        for i, level in enumerate(levels):
            y = .875-i*.042
            sidebar.plot([0, .17], [y, y], color=CMAP(.3+.7*norm(np.log(level))), lw=2)
            sidebar.text(.24, y, f"{level:g}", va="center", fontsize=10)
        y = max(.875-len(levels)*.042-.04, .36)
        sidebar.text(0, y, "READING THE MAP", fontsize=9, weight="bold")
        for dy, style, color, text in [(.055, "-", INK, "Within data hull"),
                                      (.105, "--", INK, "Extrapolated contour"),
                                      (.155, ":", TEAL, "Data hull boundary"),
                                      (.205, "--", AMBER, "Volume region")]:
            sidebar.plot([0, .17], [y-dy, y-dy], ls=style, color=color, lw=1)
            sidebar.text(.24, y-dy, text, va="center", fontsize=8)
        sidebar.scatter([.07], [y-.255], s=22, c="white", edgecolors=INK)
        sidebar.text(.24, y-.255, "Measurement", va="center", fontsize=8)
        sidebar.scatter([.07], [y-.305], marker="*", s=80, c="#B9573C")
        sidebar.text(.24, y-.305, "Source reference", va="center", fontsize=8)
        outside = [f"{r['level_cm']:g}" for r in contours["levels"] if r["status"] == "extends_beyond_map"]
        foot = "Closed contours follow the fitted surface. Dashed sections are extrapolated, not evidence of measured absence."
        fig.text(.075, .09, foot, fontsize=9, color=MUTED)
        if outside:
            fig.text(.075, .058, f"Unresolved at map limit: {', '.join(outside)} cm. These curves have not been artificially closed.",
                     fontsize=9, color=AMBER)
        else:
            fig.text(.075, .058, "Volumes refer to the ochre rectangle; the display extent is chosen independently to show closed curves.",
                     fontsize=9, color=MUTED)
        fig.text(.075, .026, "Source-centred Lambert azimuthal equal-area projection  ·  Coastlines: Natural Earth, public domain",
                 fontsize=7.5, color=MUTED)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(output/f"isopachs.{suffix}", dpi=180)
        plt.close(fig)


def save_diagnostics(output, name, report, contours, data):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        fig.subplots_adjust(left=.075, right=.97, bottom=.22, top=.77, wspace=.39)
        fig.text(.075, .93, name, fontsize=18, weight="bold")
        fig.text(.075, .85, "What supports this reconstruction?", fontsize=12, color=MUTED)
        closed = [r for r in contours["levels"] if r["status"] == "closed"]
        axes[0].plot([np.sqrt(r["area_km2"]) for r in closed], [r["level_cm"] for r in closed],
                     "o-", color=TEAL, ms=4)
        axes[0].set(yscale="log", xlabel="Square root of enclosed area (km)", ylabel="Median thickness (cm)",
                    title="01  Closed-contour areas")
        rows = report["volume_analysis"]["domain_sensitivity"]
        x = [r["domain_factor"] for r in rows]
        axes[1].plot(x, [r["selected"]["mean_km3"] for r in rows], "o-", color=TEAL, label="Selected fit")
        if all(r["alternative"] for r in rows):
            axes[1].plot(x, [r["alternative"]["mean_km3"] for r in rows], "o--", color=AMBER, label="Other thinning family")
        axes[1].set(xlabel="Integration-region width multiplier", ylabel="Expected bulk volume (km³)",
                    title="02  Sensitivity to region size", xticks=x)
        axes[1].legend(frameon=False, fontsize=8)
        nested = report["nested_validation"]
        if nested:
            lookup = {int(row): (value, censored) for row, value, censored in
                      zip(data.row_ids, data.log_value, data.censored)}
            detected = np.array([not lookup[int(row)][1] for row in nested["row_ids"]])
            actual = np.array([lookup[int(row)][0] for row in nested["row_ids"]])[detected]
            predicted = np.asarray(nested.get("median_log", nested["mean_log"]))[detected]
            lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            axes[2].plot(np.exp([lo, hi]), np.exp([lo, hi]), color=MUTED, ls="--", lw=.8)
            axes[2].scatter(np.exp(actual), np.exp(predicted), s=23, c=TEAL, alpha=.75, edgecolors="white", lw=.4)
            axes[2].set(xscale="log", yscale="log", xlabel="Withheld measurement (cm)",
                        ylabel="Predicted median (cm)", title="03  Geographic holdouts")
            axes[2].text(.03, .97, f"95% interval coverage: {nested['coverage95_observations']:.0%}",
                         transform=axes[2].transAxes, va="top", fontsize=8)
        else:
            axes[2].set_axis_off()
            axes[2].text(.05, .7, "Independent validation not run.\nUse --nested-cv to include\ngeographic holdout diagnostics.",
                         transform=axes[2].transAxes, fontsize=10)
        for ax in axes:
            ax.grid(alpha=.18, lw=.5)
            ax.tick_params(labelsize=8)
            title = ax.get_title()
            ax.set_title("")
            ax.set_title(title, fontsize=10, loc="left", pad=12)
        fig.text(.075, .05, "Numerical convergence does not validate geological extrapolation. Holdout coverage does not validate a wholly unsampled sector.",
                 fontsize=8, color=MUTED)
        for suffix in ("png", "svg"):
            fig.savefig(Path(output)/f"diagnostics.{suffix}", dpi=170)
        plt.close(fig)
