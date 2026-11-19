"""Compare the four published median fields on the same observation-area map."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from src.probabilistic_isopach.figures import STYLE
from ..core.observations import Observations
from ..region import SurveyRegion
from .global_maps import PosteriorView
from .maps import ASSET, frame_bounds, level_labels, observation_legend, observations, setup_map

FIGURE = "four-method-isopachs.png"
REGIONAL = {"stacked": "GP ensemble", "spline": "Cubic spline"}
ENSEMBLE = "directional/ensemble"
SPHERICAL = "spherical-mixture"


def read_report(path):
    return json.loads(path.read_text())


def comparison_reports(home):
    regional, ensemble, spherical = [
        read_report(home/name) for name in ("report.json", f"{ENSEMBLE}/report.json", f"{SPHERICAL}/report.json")
    ]
    if len({r["input"]["input_sha256"] for r in (regional, ensemble, spherical)}) != 1:
        raise ValueError("The four-method comparison requires identical accepted input.")
    return regional, ensemble, spherical


def saved_observations(path):
    numeric = ("x_km", "y_km", "log_value", "known_log_sd")
    rows = pd.read_csv(path, converters=dict.fromkeys(numeric, float), dtype={"censored": bool, "lower_bound": bool})
    return Observations(rows[["x_km", "y_km"]].to_numpy(), rows.log_value, rows.censored,
                        rows.known_log_sd, rows.row_id, rows.lower_bound)


def saved_regional_grid(path):
    columns = [f"{key}_median_cm" for key in REGIONAL]
    use = ["x_km", "y_km", *columns]
    table = pd.read_csv(path, usecols=use, converters=dict.fromkeys(use, float))
    ordered = table.set_index(["y_km", "x_km"]).sort_index()
    if not ordered.index.is_unique:
        raise ValueError("The regional prediction table contains repeated grid coordinates.")
    y, x = (axis.to_numpy() for axis in ordered.index.levels)
    if len(ordered) != len(y)*len(x):
        raise ValueError("The regional prediction table has an incomplete display grid.")
    fields = {key: np.log10(ordered[f"{key}_median_cm"].to_numpy().reshape(len(y), len(x))) for key in REGIONAL}
    return x, y, fields


def draw_comparison(home, destination, regional_grid=None):
    """Use the normal run's predictions or a replay of its recorded regional settings."""
    home, destination = Path(home), Path(destination)
    regional, ensemble, spherical = comparison_reports(home)
    data = saved_observations(home/"accepted_observations.csv")
    vent = regional["input"]["vent_lonlat"]
    region = SurveyRegion(data.xy)
    bounds = frame_bounds(np.vstack((region.vertices, data.xy)))
    rx, ry, fields = regional_grid if regional_grid is not None else saved_regional_grid(home/"predictions.csv")
    xx, yy = np.meshgrid(rx, ry)
    inside = region.contains(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

    with np.load(home/ENSEMBLE/"median-grids.npz") as packet:
        x, y = packet["local_x_km"], packet["local_y_km"]
        middle, = np.flatnonzero(np.isclose(packet["probabilities"], 0.5))
        directional_field = packet["local_log10_cm"][middle]
    if not np.allclose([x[0], y[0], x[-1], y[-1]], bounds, rtol=1e-10, atol=1e-6):
        raise ValueError("The directional local grid must cover the common observation frame.")
    if np.any(region.vertices.min(0) < [rx[0], ry[0]]) or np.any(region.vertices.max(0) > [rx[-1], ry[-1]]):
        raise ValueError("The regional prediction grid must contain the complete observation hull.")
    with np.load(home/SPHERICAL/"posterior.npz") as packet:
        spherical_view = PosteriorView(dict(packet), vent[1])
    xx, yy = np.meshgrid(x, y)
    sphere_field = spherical_view.central(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

    panels = []
    for key, label in REGIONAL.items():
        volume = regional["methods"][key]["regional_volume"]["expected_km3"]
        note = "" if regional["methods"][key]["regional_volume"]["converged"] else " · integral warning"
        panels.append((label, f"Hull mean · {volume:,.1f} km³{note}", rx, ry, fields[key], inside))
    map_volume = ensemble["median_surface"]["volume_km3"]
    posterior_volume = spherical["fits"]["baseline"]["volume"]["median"]
    member_checks = ensemble["coherent_fields"]["members"]
    passed = sum(m["passed"] for m in member_checks.values())
    direction_note = f" · {passed}/{len(member_checks)} fits passed" if passed < len(member_checks) else ""
    if not ensemble["median_surface"]["passed"]:
        direction_note += " · integration warning"
    sphere_note = "" if spherical["fits"]["baseline"]["diagnostics"]["passed"] else " · fit warning"
    panels += [
        ("Directional ensemble", f"Whole-Earth map integral · {map_volume:,.1f} km³{direction_note}", x, y, directional_field, None),
        ("Spherical mixture", f"Whole-Earth volume median · {posterior_volume:,.1f} km³{sphere_note}", x, y, sphere_field, None),
    ]
    peak = max(np.nanmax(z[mask] if mask is not None else z) for _, _, _, _, z, mask in panels)
    # Extend the standard 0.1 cm floor for cases with thinner accepted evidence.
    low = min(-1., np.floor(np.round(data.log_value.min()/np.log(10.), 10)))
    high = max(1., np.ceil(peak))
    norm = colors.Normalize(low, high)
    decades = np.arange(low, high+1)
    levels = np.sort(np.r_[decades, decades[:-1]+np.log10(3.)])
    inputs = ("report.json", "accepted_observations.csv", f"{ENSEMBLE}/report.json",
              f"{ENSEMBLE}/median-grids.npz", f"{SPHERICAL}/report.json", f"{SPHERICAL}/posterior.npz")
    # Retain actual plotted values so the four panel assignments can be replayed
    # from the signed fits independently of the rendering backend.
    check_points = {}
    for label, _, gx, gy, z, mask in panels:
        valid = np.isfinite(z) if mask is None else np.isfinite(z) & mask
        indices = np.flatnonzero(valid)
        row, col = np.unravel_index(indices[np.linspace(0, len(indices)-1, 5, dtype=int)], z.shape)
        check_points[label] = np.column_stack((gx[col], gy[row], z[row, col])).tolist()
    provenance = {
        "input_sha256": regional["input"]["input_sha256"],
        "sources_sha256": {name: sha256((home/name).read_bytes()).hexdigest() for name in inputs},
        "coastline_sha256": sha256(ASSET.read_bytes()).hexdigest(),
        "statistic": "pointwise median latent thickness", "panels": [p[0] for p in panels],
        "common_bounds_km": bounds.tolist(), "contour_levels_cm": (10**levels).tolist(),
        "colour_limits_cm": [10**norm.vmin, 10**norm.vmax],
        "regional_clip": "exact accepted observation hull", "spherical_display_draws": spherical_view.n,
        "check_points_x_km_y_km_log10_cm": check_points,
    }
    span = bounds[2:] - bounds[:2]
    height = np.clip(9.4*span[1]/span[0] + 1.9, 6.3, 11.0)
    hull = np.vstack((region.vertices, region.vertices[0]))
    hull_colour = "#ed974d"
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.3, height), layout="compressed",
                                 sharex=True, sharey=True)
        for ax, (label, volume, gx, gy, z, mask) in zip(axes.flat, panels, strict=True):
            ax.set_facecolor("#f3f5f4")
            image = ax.pcolormesh(gx, gy, z, cmap="viridis", norm=norm, shading="auto", zorder=1)
            if mask is not None:
                image.set_clip_path(Polygon(region.vertices, closed=True, transform=ax.transData))
                z = np.where(mask, z, np.nan)
            finite = z[np.isfinite(z)]
            visible = levels[(levels > finite.min()) & (levels < finite.max())]
            if len(visible):
                contours = ax.contour(gx, gy, z, levels=visible, colors="white", linewidths=.8,
                                      linestyles="solid", zorder=4)
                contours.set_path_effects([effects.withStroke(linewidth=1.6, foreground="#25414b", alpha=.7)])
                major = visible[np.isclose(visible, np.round(visible))]
                if len(major):
                    labels = ax.clabel(contours, levels=major, fmt=level_labels, fontsize=8, inline_spacing=4)
                    for item in labels:
                        item.set_path_effects([effects.withStroke(linewidth=2., foreground="#283c45")])
            setup_map(ax, bounds, vent)
            ax.plot(*hull.T, color=hull_colour, lw=1.1, ls=(0, (4, 2)), zorder=5,
                    path_effects=[effects.withStroke(linewidth=2.0, foreground="#413a35", alpha=.7)])
            observations(ax, data, size=14)
            ax.set_title(label, fontsize=12, loc="left", pad=15)
            ax.annotate(volume, (0, 1), xycoords="axes fraction", xytext=(0, 2.5),
                        textcoords="offset points", fontsize=9, color="#375661", va="bottom")
        for ax in axes[0]:
            ax.set_xlabel("")
        for ax in axes[:, 1]:
            ax.set_ylabel("")
        fig.suptitle(f"{regional['dataset']} · median fallout thickness", fontsize=16, fontweight="bold")
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), location="bottom", shrink=.78,
                           pad=.055, aspect=48, extend="min")
        bar.set_ticks(levels, labels=[level_labels(v) if len(levels) <= 11 or np.isclose(v, round(v)) else ""
                                     for v in levels])
        bar.set_label("Median thickness (cm, logarithmic scale)")
        observation_legend(fig, data, [Line2D([], [], color=hull_colour, ls="--", label="Observation hull")])
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination/FIGURE, dpi=170, bbox_inches="tight", pad_inches=.13,
                    metadata={"Description": json.dumps(provenance, sort_keys=True)})
        plt.close(fig)


def comparison_section(home, *, prefix=""):
    """Render the same data-backed section for a run or its published case home."""
    from .summary import metrics

    regional, ensemble, spherical = comparison_reports(Path(home))
    name = regional["dataset"]
    centre = ensemble["median_surface"]
    volumes = ensemble["coherent_fields"]["volume_km3"]
    measured = sum(centre["parts_sectors_km3"][0])/centre["volume_km3"]
    mixture = spherical["fits"]["baseline"]["volume"]
    summary = []
    for key, label in REGIONAL.items():
        model = regional["methods"][key]
        q = model["joint_volume"]["conditional_quantiles_km3"]
        summary.append((label, f"Hull mean **{model['regional_volume']['expected_km3']:,.2f}**",
                        f"{q['0.025']:,.2f}–{q['0.975']:,.2f}"))
    summary += [
        (f"[Directional ensemble]({prefix}directional/README.md)",
         f"Whole-Earth median map **{centre['volume_km3']:,.1f}** · volume median {volumes['median']:,.1f}",
         f"{volumes['lower95']:,.1f}–{volumes['upper95']:,.1f}"),
        (f"[Spherical mixture]({prefix}spherical-mixture/README.md)",
         f"Whole-Earth volume median **{mixture['median']:,.1f}**",
         f"{mixture['lower95']:,.1f}–{mixture['upper95']:,.1f}"),
    ]
    performance = metrics(regional, ensemble, spherical)
    lines = [
        "## Isopach fits", "",
        f"![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at {name}]({prefix}{FIGURE})",
        "", "All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. "
        "The regional fits stop there. The global fits also show how thickness continues into unsampled ground. "
        "Gaps inside the hull can still be poorly constrained.", "",
        "## Volume and predictive performance", "",
        "| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |",
        "|---|---|---:|---:|---:|",
    ]
    for (label, volume, interval), (_, score, coverage, passed) in zip(summary, performance, strict=True):
        score_text = "—" if score is None else f"{score:.3f}"
        coverage_text = "—" if coverage is None else f"{coverage:.1%}"
        lines.append(f"| {label} | {volume} | {interval} | {score_text}{'' if passed else ' †'} | {coverage_text} |")
    lines.extend(["", "A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out "
                      "observations inside the 95% predictive interval."
                      + (" † A geographic fit had a numerical warning or incomplete scoring."
                         if any(not r[3] for r in performance) else ""),
                  "", f"The directional map integrates pointwise median thickness. Measured interpolation triangles account for **{measured:.1%}** "
                      "of its volume. The directional volume interval comes from complete simulated deposits. "
                      "The regional intervals condition on fitted settings."
                      + (f" [Regional maps, fits and diagnostics]({prefix}README.md)" if prefix else ""),
                  "", "<details>", "<summary>Compare the two whole-Earth maps</summary>", "",
                  "| Directional ensemble | Spherical mixture |", "|---|---|",
                  f"| ![Directional ensemble median thickness over the Earth for {name}]({prefix}{ENSEMBLE}/world.png) "
                  f"| ![Spherical-mixture median thickness over the Earth for {name}]({prefix}{SPHERICAL}/world.png) |", "",
                  "Both maps use the vent longitude as their central meridian. Read the thickness key under each map.", "",
                  "</details>", ""])
    bearing = Path(home)/"directional/bearing"
    revised = read_report(bearing/"quantitative/result.json") if (bearing/"artifacts.json").is_file() else None
    if revised is not None and revised["input_sha256"] == regional["input"]["input_sha256"]:
        value = revised["median_surface"]["volume_km3"]
        q = revised["coherent_fields"]["volume_km3"]
        lines[0] = "## Four matched thickness-only fits"
        lines[1:1] = ["", "The directional control in this four-panel comparison includes a possible fan in the largest sampling gap. Its scores and volumes refer to that six-structure model."]
        lines[:0] = ["## Directional reach and archive evidence", "",
                    f"The measured-fan median map integrates to **{value:,.1f} km³**. The complete-deposit median is {q['median']:,.1f} km³, with a conditional 95% interval of {q['lower95']:,.1f}–{q['upper95']:,.1f} km³. The map shows where the predicted extent of each level changes under the recorded assumptions.", "",
                    f"![Agreement on median isopach extent]({prefix}directional/bearing/isopach-agreement.png)", "",
                    f"[Directional estimates, inspected archives and transfer sensitivities]({prefix}directional/bearing/README.md)", ""]
        checked = sorted(f for f in bearing.glob("*/validation.json") if f.parent.name!="quantitative")
        if checked:
            scored=read_report(checked[0].with_name("result.json"))
            conditional=scored["median_surface"]["volume_km3"]
            joint=scored["coherent_fields"]["volume_km3"]
            lines[3:3]=["", f"The archive sensitivity **{scored['scenario']['title']}** gives a median-map volume of **{conditional:.1f} km³**. The complete-deposit median is {joint['median']:.1f} km³ (conditional 95% interval {joint['lower95']:.1f}–{joint['upper95']:.1f} km³). This result depends on the stated transfer between local fallout and a recognizable layer."]
    return "\n".join(lines)


def add_comparison_to_run(output):
    """Put the four fits first while retaining the complete regional report below."""
    output = Path(output)
    draw_comparison(output, output)
    page = output/"README.md"
    regional = page.read_text()
    marker = "\n## Regional fit details\n"
    if marker in regional:
        regional = regional.partition(marker)[2]
    elif regional.startswith("# "):
        regional = regional.partition("\n")[2]
    name = read_report(output/"report.json")["dataset"]
    content = (f"# {name} fallout reconstruction\n\n" + comparison_section(output)
               + "\n[Volume and geographic prediction plot](overview.png)\n"
               + marker + "\n" + regional.lstrip())
    temporary = page.with_suffix(".md.tmp")
    temporary.write_text(content)
    temporary.replace(page)
