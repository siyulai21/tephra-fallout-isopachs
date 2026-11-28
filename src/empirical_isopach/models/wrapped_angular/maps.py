"""Three map scales from the same released median grid."""

import numpy as np
import matplotlib.pyplot as plt
from ...visualization.deposit_field import PolarField
from ...visualization.deposit_maps import colours, panel
from .packet import FAMILY


def three_scales(folder, data, meta, name, receipt):
    with np.load(folder / "median-grid.npz") as f:
        field = PolarField(f["radii_km"], f["log10_cm"], float(f["axis"]))
    near = np.array(receipt["near_bounds_km"])
    full = np.array(receipt["full_bounds_km"])
    half = np.sqrt(near[2] * np.max(abs(full)))
    intermediate = np.array([-half, -half, half, half])
    levels = np.array(receipt["levels_cm"])
    palette = colours(len(levels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, (bounds, title) in enumerate(
        zip((near, intermediate, full), ("Proximal", "Intermediate", "Full deposit"))
    ):
        handles, count = panel(
            axes[i],
            field,
            data,
            meta,
            FAMILY,
            None,
            bounds,
            levels,
            palette,
            i == 2,
            0.5 if i < 2 else receipt["outer_contour"]["level_cm"],
            receipt["inner_contour"]["level_cm"],
        )
        axes[i].set_title(f"{title} · {count} sites", loc="left", fontsize=11)
    fig.suptitle(name + " · wrapped angular covariance", fontsize=15, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    try:
        fig.savefig(folder / "three-scales.png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
