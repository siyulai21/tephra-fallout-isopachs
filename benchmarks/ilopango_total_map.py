"""Show the revised Figure 9 sites, including the Osa point, on all three matched fields."""

import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from .ilopango_total_airfall import INPUT, RESULT, read, digest, VENT
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.visualization.deposit_field import PolarField
from src.empirical_isopach.visualization.maps import coastlines

FAMILIES = {
    "directional": "Directional",
    "spherical-mixture": "Spherical mixture",
    "nested-isopachs": "Nested dispersal footprints",
}
REVIEWED = {"D19-L08": "L08", "D19-L10": "L10", "D19-L12": "L12", "D19-CR": "M12"}
MARKERS = {"measurement": "o", "lower_limit": "^", "upper_limit": "v", "interval": "s"}


def draw(folder, verify=False):
    raw = read(INPUT)
    data, meta = audited_data(INPUT)
    rows = [raw[i - 2] for i in data.row_ids]
    keys = [r["EvidenceID"] for r in rows]
    locs = {k: data.xy[keys.index(k)] for k in REVIEWED}
    x, y = data.xy.T
    scope = [float(x.min() - 55), float(y.min() - 55), float(x.max() + 85), float(y.max() + 55)]
    receipt = dict(
        input_sha256=digest(INPUT),
        sites=len(rows),
        overview_bounds_km=scope,
        contour_levels_cm=[0.1, 1, 10, 100],
        figures={},
        median_at_reviewed_sites_cm={},
    )
    fields = {}
    for family in FAMILIES:
        home = folder / family
        map_receipt = json.loads((home / "deposit.json").read_text())
        assert map_receipt["input_sha256"] == meta["input_sha256"]
        assert digest(home / "median-grid.npz") == map_receipt["median_grid_sha256"]
        with np.load(home / "median-grid.npz") as saved:
            field = PolarField(saved["radii_km"], saved["log10_cm"], float(saved["axis"]))
        fields[family] = field
        receipt["figures"][family] = map_receipt["median_grid_sha256"]
        receipt["median_at_reviewed_sites_cm"][family] = {
            k: float(10 ** field.at_xy(*xy)) for k, xy in locs.items()
        }
    src = folder / "all-sites-map.json"
    png = folder / "all-sites-map.png"
    if verify:
        expected = json.loads(src.read_text())
        assert expected.pop("figure_sha256") == digest(png)
        for f in FAMILIES:
            for k in REVIEWED:
                np.testing.assert_allclose(
                    receipt["median_at_reviewed_sites_cm"][f][k],
                    expected["median_at_reviewed_sites_cm"][f][k],
                    rtol=1e-9,
                )
        assert {k: v for k, v in expected.items() if k != "median_at_reviewed_sites_cm"} == {
            k: v for k, v in receipt.items() if k != "median_at_reviewed_sites_cm"
        }
        with Image.open(png) as im:
            assert json.loads(im.info["Description"])["input_sha256"] == digest(INPUT)
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.11, left=0.08, right=0.98, hspace=0.31, wspace=0.19)
    fig.suptitle("Ilopango  ·  total TBJ airfall", fontsize=19, fontweight="bold", y=0.981)
    fig.text(
        0.5,
        0.934,
        "All 168 fitted localities in the three model panels, including M12 off the Osa Peninsula",
        ha="center",
        fontsize=11,
    )
    for ax, (family, title) in zip(
        axes.flat, [(None, "Dull terrestrial readings near the vent"), *FAMILIES.items()]
    ):
        x0, y0, x1, y1 = scope if family else [-106, -52, 66, 72]
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal", facecolor="#f7f8f5")
        coastlines(ax, VENT, (x0, y0, x1, y1))
        ax.set_title(title, fontsize=12, fontweight="bold")
        if family:
            a, b = np.meshgrid(np.linspace(x0, x1, 230), np.linspace(y0, y1, 205))
            z = fields[family].at_xy(a, b)
            c = ax.contour(
                a,
                b,
                z,
                levels=[-1, 0, 1, 2],
                colors=["#13777a", "#187c83", "#135a72", "#3c536d"],
                linestyles="dashed",
                linewidths=[0.9, 1.25, 1.25, 1.3],
                zorder=2,
            )
            ax.clabel(c, fmt={-1: "0.1 cm", 0: "1 cm", 1: "10 cm", 2: "100 cm"}, fontsize=7)
        for kind, marker in MARKERS.items():
            selection = np.array([r["Kind"] == kind for r in rows])
            ax.scatter(
                x[selection],
                y[selection],
                marker=marker,
                s=17 if not family else 9,
                facecolors="#fffdf6",
                edgecolors="#464b51",
                linewidths=0.65,
                zorder=3,
            )
        ax.scatter([0], [0], marker="*", s=105, c="#fbcb67", edgecolors="#432f20", linewidths=0.9, zorder=5)
        for k, xy in locs.items():
            if (family and k != "D19-CR") or (not family and k == "D19-CR"):
                continue
            ax.scatter(*xy, s=75, facecolors="none", edgecolors="#ac3e27", linewidths=1.5, zorder=6)
            offset = {"D19-L08": (-37, 13), "D19-L10": (9, 6), "D19-L12": (10, -5), "D19-CR": (-78, 5)}[k]
            name = {
                "D19-L08": "L08  25–60",
                "D19-L10": "L10  >60",
                "D19-L12": "L12  >65",
                "D19-CR": "M12  >1? cm",
            }[k]
            ax.annotate(
                name,
                xy,
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="#86311f",
                arrowprops=dict(arrowstyle="-", color="#9c4938", lw=0.7),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=1.2),
                zorder=7,
            )
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Easting from vent (km)", fontsize=9)
        ax.set_ylabel("Northing from vent (km)", fontsize=9)
    legend = [
        Line2D(
            [],
            [],
            marker=m,
            color="none",
            markeredgecolor="#444",
            markerfacecolor="#fffdf6",
            markersize=6,
            label=k.replace("_", " ").title(),
        )
        for k, m in MARKERS.items()
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.034), fontsize=9
    )
    fig.text(
        0.5,
        0.018,
        "Dashed contours show conditional median thickness. Red rings identify corrected source readings.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(png, dpi=160, metadata={"Description": json.dumps(receipt, sort_keys=True)})
    plt.close(fig)
    receipt["figure_sha256"] = digest(png)
    src.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=RESULT)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    draw(args.folder, args.verify)
