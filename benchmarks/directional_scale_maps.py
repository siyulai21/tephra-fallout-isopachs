"""Compare the same doubling contours in three nested geographic windows."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.visualization.deposit_field import PolarField
from src.empirical_isopach.visualization.deposit_maps import draw_land, regular_axes, frames
from src.empirical_isopach.visualization.contour_limits import local_half_width

ARCHIVE = Path("archive/2026-11-25-directional-geography")
LEVELS = 0.5 * 2.0 ** np.arange(16)


def read_grid(path):
    with np.load(path) as z:
        return PolarField(z["radii_km"], z["log10_cm"], float(z["axis"]))


def windows(receipts):
    ext = np.array([r["full_bounds_km"] for r in receipts])
    full = np.r_[ext[:, :2].min(axis=0), ext[:, 2:].max(axis=0)]
    near = np.array(receipts[0]["near_bounds_km"])
    h = np.sqrt(near[2] * np.max(abs(full)))
    return (near, np.array([-h, -h, h, h]), full)


def panel(ax, field, data, meta, box, inner=()):
    x, y = regular_axes(box, 425)
    draw_land(ax, box, meta["vent_lonlat"], 220)
    z = field.on_grid(x, y)
    levels = LEVELS[(np.log10(LEVELS) > np.nanmin(z)) & (np.log10(LEVELS) < np.nanmax(z))]
    if len(levels):
        colors = ["#12678c" if v <= 2 else "#4f7466" if v < 32 else "#ad4e26" for v in levels]
        c = ax.contour(x, y, z, levels=np.log10(levels), colors=colors, linewidths=0.85)
        ax.clabel(
            c, c.levels[:: max(1, len(levels) // 3)], fmt=lambda l: f"{10**l:g}", fontsize=6, inline_spacing=2
        )
    interval = np.isfinite(data.interval_log_upper)
    for m, marker in (
        (~data.censored, "o"),
        (data.lower_censored & ~interval, "^"),
        (data.censored & ~data.lower_censored & ~interval, "v"),
        (interval, "s"),
    ):
        ax.scatter(*data.xy[m].T, marker=marker, s=8, c="#323d46", lw=0, alpha=0.62, zorder=4)
    ax.plot(0, 0, "*", mfc="#ffecac", mec="#613b2a", ms=9, zorder=6)
    for window in inner:
        ax.add_patch(
            Rectangle(
                window[:2],
                *(window[2:] - window[:2]),
                fill=False,
                edgecolor="#785e83",
                lw=0.9,
                ls=":",
                zorder=5,
            )
        )
    in_view = np.all((data.xy >= box[:2]) & (data.xy <= box[2:]), axis=1)
    ax.text(
        0.025,
        0.025,
        f"{int(in_view.sum())} fitted locations",
        transform=ax.transAxes,
        va="bottom",
        fontsize=7,
        color="#303f40",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=2),
    )
    ax.set(aspect="equal", xlim=box[::2], ylim=box[1::2])
    ax.tick_params(labelsize=7, length=2)


def plate(items, output, title):
    fig, axes = plt.subplots(
        len(items), 3, figsize=(12.6, 3.95 * len(items) + 0.9), layout="constrained", squeeze=False
    )
    for row, (name, field, d, m, boxes) in enumerate(items):
        for col, (ax, box) in enumerate(zip(axes[row], boxes, strict=True)):
            panel(ax, field, d, m, box, boxes[:col])
            if not row:
                ax.set_title(("Proximal", "Intermediate", "Full deposit")[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(name + "\nNorthing (km)", fontsize=9)
            if row == len(items) - 1:
                ax.set_xlabel("Easting from vent (km)", fontsize=8)
    fig.suptitle(title, fontsize=15)
    fig.legend(
        handles=[
            Line2D([], [], ls="", marker=k, color="#48555a", label=s, ms=5)
            for k, s in (
                ("o", "Measurement"),
                ("^", "Lower limit"),
                ("v", "Upper limit"),
                ("s", "Interval"),
                ("*", "Vent"),
            )
        ]
        + [Line2D([], [], ls=":", color="#785e83", label="Geographic windows enlarged at left")],
        loc="outside lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    fig.savefig(output, dpi=132)
    plt.close(fig)


def figures(scratch, out, initial=False):
    out.mkdir(exist_ok=True, parents=True)
    gallery = []
    for r in datasets():
        slug = r["slug"]
        home = Path("eruptions") / slug / "results/directional"
        old = ARCHIVE / slug if (ARCHIVE / slug / "median-grid.npz").exists() else home
        rec = json.loads((old / "deposit.json").read_text())
        data, meta = audited_data(r["path"])
        assert rec["input_sha256"] == meta["input_sha256"]
        oldfield = read_grid(old / "median-grid.npz")
        if initial:
            gallery.append((r["name"], oldfield, data, meta, windows([rec])))
            if len(gallery) == 3:
                plate(gallery, out / (slug + ".png"), "Published directional isopachs · thickness in cm")
                gallery = []
            continue
        new = scratch / slug
        current = json.loads((new / "deposit.json").read_text())
        newfield = read_grid(new / "median-grid.npz")
        near = local_half_width(data) * np.array([-1.0, -1.0, 1.0, 1.0])
        bounds = windows(
            [dict(full_bounds_km=frames(f, data, 0.5)[0], near_bounds_km=near) for f in (oldfield, newfield)]
        )
        items = [
            ("Previous directional fit", oldfield, data, meta, bounds),
            ("Revised directional fit", newfield, data, meta, bounds),
        ]
        plate(items, out / (slug + ".png"), r["name"] + " · identical observations and scale in each column")
    if gallery:
        plate(gallery, out / "remaining.png", "Published directional isopachs · thickness in cm")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", "--scratch", dest="scratch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", action="store_true")
    a = p.parse_args()
    figures(a.scratch, a.output, a.initial)
