"""Plot B-Tm source classifications from the audited ledgers."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.empirical_isopach.visualization.maps import coast_rings

AUDIT = Path("data/audits/B-Tm")
STYLES = {
    "Measured bed": ("o", "#196b74", 23),
    "Lower bound": ("^", "#b34e24", 49),
    "Upper bound": ("v", "#9d3e71", 49),
    "Reported range": ("s", "#93772a", 29),
    "Locality thickness": ("P", "#b57415", 55),
    "Occurrence or patch": ("D", "#77638d", 27),
    "Unresolved record": ("x", "#6e7377", 25),
}


def read_points():
    groups = {"measurement": "Measured bed", "lower_limit": "Lower bound",
              "upper_limit": "Upper bound", "range": "Reported range",
              "locality_range": "Reported range", "locality_thickness": "Locality thickness"}
    unresolved = {"unresolved", "location_conflict", "attribution_conflict", "unresolved_workbook"}
    points = []
    for name, key in (("evidence.csv", "evidence_kind"), ("additional-evidence.csv", "evidence")):
        with (AUDIT/name).open() as stream:
            for row in csv.DictReader(stream):
                if row["latitude"] and row["longitude"]:
                    kind = row[key]
                    label = "Unresolved record" if kind in unresolved else groups.get(kind, "Occurrence or patch")
                    points.append((float(row["longitude"]), float(row["latitude"]), label))
    return points


def panel(ax, points, bounds):
    west, east, south, north = bounds
    coast = [ring for ring in coast_rings() if ring[:, 0].max() >= west and ring[:, 0].min() <= east
             and ring[:, 1].max() >= south and ring[:, 1].min() <= north]
    ax.add_collection(LineCollection(coast, colors="#9aa8ad", linewidths=.55))
    for label, (shape, color, size) in reversed(STYLES.items()):
        xy = np.array([(x, y) for x, y, group in points if group == label]).reshape(-1, 2)
        edges = {} if shape == "x" else {"edgecolors": "white", "linewidths": .45}
        ax.scatter(*xy.T, marker=shape, c=color, s=size, alpha=.92, **edges)
    ax.scatter(128.057, 42.006, c="#20282a", marker="*", s=130, edgecolors="white", linewidths=.6, zorder=9)
    ax.set(xlim=(west, east), ylim=(south, north), xlabel="Longitude (°E)",
           ylabel="Latitude (°N)", aspect=1/np.cos(np.deg2rad((south+north)/2)))
    ax.ticklabel_format(useOffset=False)
    ax.grid(color="#c4ced0", alpha=.4, linewidth=.5)
    ax.set_axisbelow(True)


def render(path):
    points = read_points()
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.9), gridspec_kw={"width_ratios": [1.72, 1]})
    zoom = (127.98, 128.64, 41.94, 42.28)
    panel(axes[0], points, (125.7, 146, 34.8, 46.2))
    panel(axes[1], points, zoom)
    axes[0].add_patch(Rectangle((zoom[0], zoom[2]), zoom[1]-zoom[0], zoom[3]-zoom[2],
                                fill=False, edgecolor="#354246", linestyle="--", linewidth=.8))
    axes[0].set_title("East Asian records with locations")
    axes[1].set_title("Proximal exposures")
    for ax, label, point, offset in [
        (axes[0], "Sihailongwan", (126.60, 42.2833), (4, 12)),
        (axes[0], "Kushu", (141.036944, 45.431944), (-29, 8)),
        (axes[0], "Suigetsu", (135.883813, 35.583478), (4, -13)),
        (axes[1], "Tianwenfeng", (128.066667, 42.025833), (4, 7)),
        (axes[1], "Dongfanghong", (128.318139, 42.192278), (5, 5)),
        (axes[1], "Yuanchi", (128.539, 41.998), (-26, -14)),
    ]:
        ax.annotate(label, point, xytext=offset, textcoords="offset points", fontsize=8, color="#30373a")
    present = {label for _, _, label in points}
    handles = [Line2D([], [], marker=shape, linestyle="none", color=color, label=label)
               for label, (shape, color, _) in STYLES.items() if label in present]
    handles.append(Line2D([], [], marker="*", linestyle="none", color="#20282a", label="Vent"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.2, top=.93, wspace=.23)
    fig.savefig(path, dpi=175, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    render(parser.parse_args().output)
