"""Compare the same doubling contours in three nested geographic windows."""

import argparse
import json
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from .publication import result_home
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.visualization.deposit_field import PolarField, digest
from src.empirical_isopach.visualization.deposit_maps import draw_land, regular_axes, frames
from src.empirical_isopach.visualization.contour_limits import local_half_width

ARCHIVE = Path("archive/2026-11-25-directional-geography")
LEVELS = 0.5 * 2.0 ** np.arange(16)


def preceding_control(record, scratch):
    """Reuse or render the preceding extent model on this exact fitting input."""
    folder = result_home(record) / "directional"
    identity = digest(record["path"])
    cached = scratch / "preceding-directional-controls" / record["slug"]
    for candidate in (ARCHIVE / record["slug"], folder, cached):
        page = candidate / "deposit.json"
        if not page.is_file():
            continue
        receipt = json.loads(page.read_text())
        sources = receipt["sources_sha256"]
        if (receipt["input_sha256"] == identity and "extent/report.json" in sources
                and "multiscale/report.json" not in sources
                and all(digest(folder / name) == h for name, h in sources.items())
                and digest(candidate / "median-grid.npz") == receipt["median_grid_sha256"]):
            return candidate
    from src.empirical_isopach.visualization.deposit import render_deposit
    data, meta = audited_data(record["path"])
    shutil.copytree(folder / "extent", cached / "extent", dirs_exist_ok=True)
    render_deposit(folder, "directional", data, meta, record["name"], output=cached, write_page=False)
    return cached


def read_grid(path):
    with np.load(path) as z:
        return PolarField(z["radii_km"], z["log10_cm"], float(z["axis"]))


def windows(receipts):
    ext = np.array([r["full_bounds_km"] for r in receipts])
    full = np.r_[ext[:, :2].min(axis=0), ext[:, 2:].max(axis=0)]
    near = np.array(receipts[0]["near_bounds_km"])
    h = np.sqrt(near[2] * np.max(abs(full)))
    return (near, np.array([-h, -h, h, h]), full)


def panel(ax, field, data, meta, box, inner=(), model=None):
    x, y = regular_axes(box, 425)
    draw_land(ax, box, meta["vent_lonlat"], 220)
    z = field.on_grid(x, y)
    if model is not None:
        xy = np.column_stack([v.ravel() for v in np.meshgrid(x, y)])
        valid = model.domain.rho(xy) <= model.domain.rho_max
        z = np.full(len(xy), np.nan)
        z[valid] = model.central(xy[valid])
        z = z.reshape(len(y), len(x))
    levels = LEVELS[(np.log10(LEVELS) > np.nanmin(z)) & (np.log10(LEVELS) < np.nanmax(z))]
    if len(levels):
        colors = ["#12678c" if v <= 2 else "#4f7466" if v < 32 else "#ad4e26" for v in levels]
        c = ax.contour(x, y, z, levels=np.log10(levels), colors=colors, linewidths=0.85,
                       linestyles="dashed")
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
    for row, item in enumerate(items):
        name, field, d, m, boxes = item[:5]
        model = item[5] if len(item) > 5 else None
        for col, (ax, box) in enumerate(zip(axes[row], boxes, strict=True)):
            panel(ax, field, d, m, box, boxes[:col], model)
            if not row:
                ax.set_title(("Proximal", "Intermediate", "Full deposit")[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(name + "\nNorthing (km)", fontsize=9)
            if row == len(items) - 1:
                ax.set_xlabel("Easting from vent (km)", fontsize=8)
    fig.suptitle(title, fontsize=15)
    observations = [item[2] for item in items]
    point_label = "Point (includes reported minima)" if any(d.reported_minimum.any() for d in observations) else "Measurement"
    symbols = [("o", point_label)]
    if any((d.lower_censored & ~d.interval).any() for d in observations):
        symbols.append(("^", "Lower limit"))
    if any((d.censored & ~d.lower_censored).any() for d in observations):
        symbols.append(("v", "Upper limit"))
    if any(d.interval.any() for d in observations):
        symbols.append(("s", "Interval"))
    symbols.append(("*", "Vent"))
    fig.legend(
        handles=[Line2D([], [], ls="", marker=marker, color="#48555a", label=label, ms=5)
                 for marker, label in symbols]
        + [Line2D([], [], ls=":", color="#785e83", label="Windows enlarged at left"),
           Line2D([], [], ls="--", color="#4f7466", label="Median field; support shown on case plates")],
        loc="outside lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    fig.savefig(output, dpi=132)
    plt.close(fig)


def figures(scratch, out, initial=False, baseline=None):
    out.mkdir(exist_ok=True, parents=True)
    gallery = []
    for r in datasets():
        slug = r["slug"]
        old = baseline / slug if baseline else preceding_control(r, scratch)
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
        assert current["input_sha256"] == meta["input_sha256"]
        newfield = read_grid(new / "median-grid.npz")
        near = local_half_width(data) * np.array([-1.0, -1.0, 1.0, 1.0])
        bounds = windows(
            [dict(full_bounds_km=frames(f, data, 0.5)[0], near_bounds_km=near) for f in (oldfield, newfield)]
        )
        items = [
            ("Previous directional fit", oldfield, data, meta, bounds),
            ("Revised directional fit", newfield, data, meta, bounds),
        ]
        if baseline:
            from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField
            items = [(*item, DirectionalField.load(home / "multiscale/posterior.npz"))
                     for item, home in zip(items, (old, new), strict=True)]
        plate(items, out / (slug + ".png"), r["name"] + " · identical observations and scale in each column")
        if baseline:
            review_receipt(out, slug, meta, new / "multiscale/posterior.npz", old, bounds)
    if gallery:
        plate(gallery, out / "remaining.png", "Published directional isopachs · thickness in cm")


def review_receipt(output, slug, meta, current, old, boxes):
    receipt = dict(input_sha256=meta["input_sha256"], posterior_sha256=digest(current),
                   previous_posterior_sha256=digest(old / "multiscale/posterior.npz"),
                   windows_km=[b.tolist() for b in boxes], cartesian_long_side=425,
                   contour_evaluation="direct posterior median", figure_sha256=digest(output / (slug + ".png")))
    (output / (slug + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")


def fitted_figure(job):
    """A direct three-scale review can precede costly whole-volume integration."""
    record, work, output, baseline = job
    from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField
    from src.empirical_isopach.visualization.deposit_field import sample_field

    slug = record["slug"]
    data, meta = audited_data(record["path"])
    current = work / slug / "full/field.npz"
    old = baseline / slug
    models = [DirectionalField.load(old / "multiscale/posterior.npz"), DirectionalField.load(current)]
    fields = [read_grid(old / "median-grid.npz"), sample_field(models[1], data, meta["vent_lonlat"][1])]
    near = local_half_width(data) * np.array([-1., -1., 1., 1.])
    boxes = windows([dict(full_bounds_km=frames(f, data, .5)[0], near_bounds_km=near) for f in fields])
    items = [(label, f, data, meta, boxes, m) for label, f, m in
             zip(("Previous v5 directional", "Revised v6 directional"), fields, models, strict=True)]
    plate(items, output / (slug + ".png"), record["name"] + " · direct posterior medians on identical windows")
    review_receipt(output, slug, meta, current, old, boxes)
    print(slug, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--release", "--scratch", dest="scratch", type=Path)
    inputs.add_argument("--work", type=Path, help="Plot completed full-fit packets directly")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", action="store_true")
    p.add_argument("--baseline", type=Path, help="Archived matched multiscale release")
    p.add_argument("--jobs", type=int, default=2)
    a = p.parse_args()
    if a.work:
        if a.baseline is None:
            p.error("--work requires a matched --baseline")
        a.output.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(a.jobs) as pool:
            list(pool.map(fitted_figure, [(r, a.work, a.output, a.baseline) for r in datasets()]))
    else:
        figures(a.scratch, a.output, a.initial, a.baseline)
