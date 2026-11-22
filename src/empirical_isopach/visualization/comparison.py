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

from src.empirical_isopach.visualization.style import STYLE
from ..core.observations import Observations
from ..region import SurveyRegion
from .global_maps import PosteriorView
from .maps import ASSET, frame_bounds, level_labels, observation_legend, observations, setup_map

FIGURE = "four-method-evidence.png"
REGIONAL = {"stacked": "GP ensemble", "spline": "Cubic spline"}
DIRECTIONAL = "directional/azimuth"
DIRECTIONAL_LABEL = "Uncertain fan direction"
SPHERICAL = "spherical-mixture"


def read_report(path):
    return json.loads(path.read_text())


def comparison_reports(home):
    regional, ensemble, spherical = [
        read_report(home/name) for name in ("report.json", f"{DIRECTIONAL}/report.json", f"{SPHERICAL}/report.json")
    ]
    if ensemble["protocol"] != "uncertain-azimuth-v1":
        raise ValueError("The principal directional report must use uncertain fan direction.")
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

    with np.load(home/DIRECTIONAL/"median-grids.npz") as packet:
        x, y = packet["local_x_km"], packet["local_y_km"]
        directional_field = packet["local_log10_cm"]
    if not np.allclose([x[0], y[0], x[-1], y[-1]], bounds, rtol=1e-10, atol=1e-6):
        raise ValueError("The directional local grid must cover the common observation frame.")
    if np.any(region.vertices.min(0) < [rx[0], ry[0]]) or np.any(region.vertices.max(0) > [rx[-1], ry[-1]]):
        raise ValueError("The regional prediction grid must contain the complete observation hull.")
    from ..models.mass_mixture.bearings.field import PROTOCOL
    from ..models.mass_mixture.bearings.maps import BearingView
    bearing = spherical.get("protocol")==PROTOCOL
    with np.load(home/SPHERICAL/"posterior.npz") as packet:
        spherical_view = BearingView(dict(packet),data,vent[1],spherical["bearing"]["map_draw_check"]["draws_per_chain"]) if bearing else PosteriorView(dict(packet),vent[1])
    xx, yy = np.meshgrid(x, y)
    sphere_field = spherical_view.central(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

    panels = []
    for key, label in REGIONAL.items():
        volume = regional["methods"][key]["regional_volume"]["expected_km3"]
        note = "" if regional["methods"][key]["regional_volume"]["converged"] else " · integral warning"
        item = regional["methods"][key]
        if volume > item["joint_volume"]["conditional_quantiles_km3"]["0.975"]:
            detail = f"Hull median map · {item['regional_volume']['median_surface_km3']:,.1f} km³ · unstable mean"
        else:
            detail = f"Hull mean · {volume:,.1f} km³{note}"
        panels.append((label, detail, rx, ry, fields[key], inside))
    volume = ensemble["volume"]
    map_volume = volume["median_map_km3"]
    posterior_volume = spherical["fits"]["baseline"]["volume"]["median"]
    member_checks = ensemble["members"]
    passed = sum(m["diagnostics"]["passed"] for m in member_checks.values())
    direction_note = f" · {passed}/{len(member_checks)} fits passed" if passed < len(member_checks) else ""
    checks = volume["numerical"]
    if not checks["spatial_passed"]:
        direction_note += " · integration warning"
    elif not checks["split_map_passed"]:
        direction_note += " · map split warning"
    sphere_note = "" if spherical["fits"]["baseline"]["diagnostics"]["passed"] else " · fit warning"
    panels += [
        (DIRECTIONAL_LABEL, f"Whole-Earth map integral · {map_volume:,.1f} km³{direction_note}", x, y, directional_field, None),
        ("Spherical mixture", f"Whole-Earth volume median · {posterior_volume:,.1f} km³{sphere_note}", x, y, sphere_field, None),
    ]
    peak = max(np.nanmax(z[mask] if mask is not None else z) for _, _, _, _, z, mask in panels)
    # Extend the standard 0.1 cm floor for cases with thinner accepted evidence.
    low = min(-1., np.floor(np.round(data.log_value.min()/np.log(10.), 10)))
    high = max(1., np.ceil(peak))
    norm = colors.Normalize(low, high)
    decades = np.arange(low, high+1)
    levels = np.sort(np.r_[decades, decades[:-1]+np.log10(3.)])
    inputs = ("report.json", "accepted_observations.csv", f"{DIRECTIONAL}/report.json",
              f"{DIRECTIONAL}/median-grids.npz", f"{SPHERICAL}/report.json", f"{SPHERICAL}/posterior.npz")
    # Retain actual plotted values so the four panel assignments can be replayed
    # from the signed fits independently of the rendering backend.
    from ..models.directional.azimuth.primary import read_field
    from ..models.directional.azimuth.maps import supported_lines
    measured = read_field(home/DIRECTIONAL,data,ensemble)
    inputs += (f"{DIRECTIONAL}/fields.npz",)
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
        "spherical_protocol": spherical["protocol"], "spherical_volume_km3": posterior_volume,
        "directional_protocol": ensemble["protocol"], "directional_median_map_km3": map_volume,
        "directional_solid": "exact-site bracket in a measured triangle and posterior crossing probability >= 0.9",
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
                                      linestyles="dashed" if label==DIRECTIONAL_LABEL or (bearing and label=="Spherical mixture") else "solid", zorder=4)
                if label==DIRECTIONAL_LABEL:
                    supported_lines(ax,data,measured,gx,gy,z,visible,["white"]*len(visible))
                contours.set_path_effects([effects.withStroke(linewidth=1.6, foreground="#25414b", alpha=.7)])
                major = visible[np.isclose(visible, np.round(visible))]
                if len(major):
                    labels = ax.clabel(contours, levels=major, fmt=level_labels, fontsize=8, inline_spacing=4)
                    for item in labels:
                        item.set_path_effects([effects.withStroke(linewidth=2., foreground="#283c45")])
            setup_map(ax, bounds, vent)
            ax.plot(*hull.T, color=hull_colour, lw=1.1, ls=(0, (4, 2)), zorder=5,
                    path_effects=[effects.withStroke(linewidth=2.0, foreground="#413a35", alpha=.7)])
            observations(ax, data, size=14, meta=regional["input"])
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
        observation_legend(fig, data, [
            Line2D([], [], color=hull_colour, ls="--", label="Observation hull"),
            Line2D([], [], color="#354f5a", lw=2., label="Directional · measured bracket"),
            Line2D([], [], color="#354f5a", ls="--", label="Directional · continuation")], meta=regional["input"])
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination/FIGURE, dpi=170, bbox_inches="tight", pad_inches=.13,
                    metadata={"Description": json.dumps(provenance, sort_keys=True)})
        plt.close(fig)


def directional_sensitivities(home, ensemble, prefix):
    """Tabulate the source evidence and fan assumption for each comparable sensitivity."""
    bearing = Path(home) / "directional/bearing"
    measured = read_report(bearing/"quantitative/result.json") if (bearing/"artifacts.json").is_file() else None
    if measured and measured["input_sha256"] != ensemble["input"]["input_sha256"]:
        measured = None
    scenarios = []
    archive = ensemble["volume"].get("archive_conditioned")
    if archive:
        scenario = ensemble["archive"]["mapped_scenario"]
        title = scenario.replace("_", " ")
        if measured and (bearing/scenario/"result.json").is_file():
            candidate = read_report(bearing/scenario/"result.json")
            if (candidate["input_sha256"] == measured["input_sha256"]
                    and candidate["archive_inventory_sha256"] == ensemble["archive_evidence_sha256"]):
                title = candidate["scenario"]["title"]
        scenarios.append(("Uncertain fan direction", title, archive["median_map_km3"], archive["coherent_volume_km3"]))
    if measured:
        scenarios.append(("Measured fan directions", "Thicknesses and bounds", measured["median_surface"]["volume_km3"],
                          measured["coherent_fields"]["volume_km3"]))
        for validation in sorted(bearing.glob("*/validation.json")):
            if validation.parent.name != "quantitative":
                result = read_report(validation.with_name("result.json"))
                if (result["input_sha256"] == measured["input_sha256"]
                        and result["archive_inventory_sha256"] == measured["archive_inventory_sha256"]):
                    scenarios.append(("Measured fan directions", result["scenario"]["title"],
                                      result["median_surface"]["volume_km3"], result["coherent_fields"]["volume_km3"]))
    if not scenarios:
        return []
    heading = "Directional assumptions and inspected sites" if archive or len(scenarios) > 1 else "Directional assumptions"
    description = "The principal fit above uses quantitative thicknesses and bounds. The measured-fan model extends fallout along sampled bearings."
    if ensemble["input"].get("interpreted_upper_limits"):
        description = "Both fits include the same interpreted archive limits. The measured-fan sensitivity fixes the directions of the continuation to the sampled bearings."
    if archive or len(scenarios) > 1:
        description += " A visible-bed scenario also asks how much ash would have been preserved and recognized at inspected sites."
    lines = [f"## {heading}", "", description, "",
              "| Fan direction | Evidence assumption | Median-map volume (km³) | Complete-deposit median (km³) | Conditional 95% interval (km³) |",
              "|---|---|---:|---:|---:|"]
    for method, assumption, integral, joint in scenarios:
        lines.append(f"| {method} | {assumption} | {integral:,.1f} | {joint['median']:,.1f} | "
                     f"{joint['lower95']:,.1f}–{joint['upper95']:,.1f} |")
    if measured:
        lines += ["", "<details>", "<summary>View agreement between the directional sensitivities</summary>", "",
                  f"![Agreement on median isopach extent]({prefix}directional/bearing/isopach-agreement.png)", "",
                  "</details>", "", f"[Measured-fan maps, held-out tests and site evidence]({prefix}directional/bearing/README.md)", ""]
    return lines


def nested_comparison(home, prefix=""):
    p = Path(home)/"nested-isopachs/report.json"
    if not p.is_file():
        return None
    model = read_report(p)
    if model.get("protocol") == "nested-corridor-reach-v4":
        regional = read_report(Path(home)/"report.json")
        if model["input_sha256"] != regional["input"]["input_sha256"]:
            raise ValueError("Nested and regional results use different accepted observations")
        lo, med, hi = model["volume_km3"]
        warning = "" if all(model["numerics"].values()) else " **(provisional)**"
        passed = sum(f["passed"] for f in model["folds"])
        total = len(model["folds"])
        cov = model["exact_site_coverage95"]["fraction"]
        coverage = "—" if cov is None else f"{cov:.1%}"
        row = (f"| [Bayesian nested · corridor]({prefix}nested-isopachs/README.md) | Whole-Earth volume median **{med:,.1f}** "
               f"| {lo:,.1f}–{hi:,.1f}{warning} | {model['geographic_score']:.3f}{' †' if passed < total else ''} | {coverage} |")
        inside, far, _ = model["allocation_fraction"]
        note = (f"The nested corridor assigns a median {inside[1]:.1%} of bulk volume to the retained exact-site triangles "
                f"and {far[1]:.1%} beyond the farthest exact site. Its median map integrates to "
                f"{model['median_map_volume_km3']:,.1f} km³. {passed}/{total} buffered fits passed their numerical checks. "
                f"[Whole-area sensitivity]({prefix}nested-whole-area/README.md) · "
                f"[Paired median isopachs]({prefix}nested-isopachs/paired-isopachs.png)")
        if prefix:
            note += f" · [Five-family volume and geographic prediction plot]({prefix}overview.png)"
        return row, note
    if model["input"]["input_sha256"] != read_report(Path(home)/"report.json")["input"]["input_sha256"]:
        raise ValueError("Nested and regional results use different accepted observations")
    lo,med,hi = model["conditional_volume_km3"]
    warning = "" if model["population_check_passed"] and model["area_check"][-1]["passed"] else " **(provisional)**"
    passed = model["geographic_folds_passed"] == 4
    cov = model["exact_site_coverage95"]["fraction"]
    coverage = "—" if cov is None else f"{cov:.1%}"
    row = (f"| [Bayesian nested isopachs]({prefix}nested-isopachs/README.md) | Whole-Earth volume median **{med:,.1f}** "
           f"| {lo:,.1f}–{hi:,.1f}{warning} | {model['geographic_score']:.3f}{'' if passed else ' †'} | {coverage} |")
    note = (f"[Nested median isopachs and the area–thickness curve]({prefix}nested-isopachs/isopachs.png) · "
            f"[Nested whole-Earth map]({prefix}nested-isopachs/world.png)")
    if (p.parent/"bearing-study/README.md").is_file():
        note += f" · [Nested sensitivity in poorly sampled bearings]({prefix}nested-isopachs/bearing-study/README.md)"
    for directory,label in (("directional","Uncertain gap amount"),("spherical-mixture","Soft source displacement")):
        if (Path(home)/directory/"sensitivity-study/README.md").is_file():
            note += f" · [{label}]({prefix}{directory}/sensitivity-study/README.md)"
    if "terminal_continuation" in model:
        budget = model["spatial_volume_budget"]["fraction_by_part"]
        note += (f"\n\nThe nested estimate puts a median {budget[0][1]:.1%} of the fallout volume inside the "
                 f"retained exact-site triangles and {budget[2][1]:.1%} beyond the farthest exact measurement. "
                 f"{model['geographic_folds_passed']}/4 geographic refits passed the independent-population check.")
    return row,note


def comparison_section(home, *, prefix=""):
    """Render the same data-backed section for a run or its published case home."""
    from .summary import metrics

    regional, ensemble, spherical = comparison_reports(Path(home))
    name = regional["dataset"]
    centre = ensemble["volume"]
    volumes = centre["coherent_volume_km3"]
    measured = sum(centre["median_map_parts_by_sector_km3"][0])/centre["median_map_km3"]
    spherical_fit = spherical["fits"]["baseline"]
    mixture = spherical_fit["volume"]
    spherical_interval = f"{mixture['lower95']:,.1f}–{mixture['upper95']:,.1f}"
    if not spherical_fit["diagnostics"]["passed"]:
        spherical_interval += " **(provisional)**"
    summary = []
    for key, label in REGIONAL.items():
        model = regional["methods"][key]
        q = model["joint_volume"]["conditional_quantiles_km3"]
        vol = model["regional_volume"]
        description = f"Hull mean **{vol['expected_km3']:,.2f}**"
        if vol["expected_km3"] > q["0.975"]:
            description = (f"Hull median map **{vol['median_surface_km3']:,.2f}** · volume median {q['0.5']:,.2f}"
                           f" · unstable mean {vol['expected_km3']:.2g}")
        summary.append((label, description, f"{q['0.025']:,.2f}–{q['0.975']:,.2f}"))
    summary += [
        (f"[Directional ensemble · uncertain fan direction]({prefix}directional/README.md)",
         f"Whole-Earth median map **{centre['median_map_km3']:,.1f}** · volume median {volumes['median']:,.1f}",
         f"{volumes['lower95']:,.1f}–{volumes['upper95']:,.1f}"),
        (f"[Spherical mixture]({prefix}spherical-mixture/README.md)",
         f"Whole-Earth volume median **{mixture['median']:,.1f}**",
         spherical_interval),
    ]
    performance = metrics(regional, ensemble, spherical)
    lines = [
        "## Isopach fits", "",
        f"![Median thickness contours for the GP ensemble, cubic spline, directional ensemble and spherical mixture at {name}]({prefix}{FIGURE})",
        "", "All four maps use the same frame, contour levels and thickness scale. Thickness below the first colour-bar tick shares the lowest shade. The dashed line encloses the observation hull. "
        "The regional fits stop there. The global fits also show how thickness continues into unsampled ground. "
        + ("On the directional map, solid white segments connect thicknesses bracketed by measured sites in a supported triangle. White dashes on the global maps show conditional isopachs." if spherical.get("bearing") else "On the directional map, white dashes show conditional continuation. Solid white segments connect thicknesses bracketed by measured sites in a supported triangle.") + " Gaps inside the hull can still be poorly constrained.", "",
        "## Volume and predictive performance", "",
        "| Method | Bulk volume (km³) | Conditional 95% interval (km³) | Held-out score ↓ | 95% coverage |",
        "|---|---|---:|---:|---:|",
    ]
    for (label, volume, interval), (_, score, coverage, passed) in zip(summary, performance, strict=True):
        score_text = "—" if score is None else f"{score:.3f}"
        coverage_text = "—" if coverage is None else f"{coverage:.1%}"
        lines.append(f"| {label} | {volume} | {interval} | {score_text}{'' if passed else ' †'} | {coverage_text} |")
    nested = nested_comparison(home,prefix)
    corridor = nested and read_report(Path(home)/"nested-isopachs/report.json").get("model_role") == "principal"
    if corridor:
        lines[2:2] = [
            f"![Principal nested corridor median isopachs for {name}]({prefix}nested-isopachs/isopachs.png)", "",
            "On the nested map, solid contour segments have a local thickness bracket and at least 90% posterior crossing probability. Dashes show conditional continuation. "
            f"[Corridor estimate, support and uncertainties]({prefix}nested-isopachs/README.md)", ""]
    if nested:
        lines.append(nested[0])
        lines.extend(["",nested[1]])
    lines.extend(["", "A lower negative log predictive score means better prediction across withheld regions. Each region gets equal weight. Coverage is the share of exact held-out "
                      "observations inside the 95% predictive interval."
                      + (" † A geographic fit had a numerical warning or incomplete scoring."
                         if any(not r[3] for r in performance) or (nested and " †" in nested[0]) else ""),
                  "", f"The directional map integrates pointwise median thickness. Measured interpolation triangles account for **{measured:.1%}** "
                      "of its volume. The directional volume interval comes from "
                      + ("complete posterior deposit draws. " if corridor else "complete simulated deposits. ")
                      + ("The spherical mixture reports the median integrated volume across broad and uncertain fan continuations. " if spherical.get("bearing") else "") + "The regional intervals condition on fitted settings."
                      + (f" [Regional maps, fits and diagnostics]({prefix}README.md)" if prefix else ""),
                  "", "<details>", "<summary>Compare the two whole-Earth maps</summary>", "",
                  "| Uncertain fan direction | Spherical mixture |", "|---|---|",
                  f"| ![Uncertain-fan-direction median thickness over the Earth for {name}]({prefix}{DIRECTIONAL}/world.png) "
                  f"| ![Spherical-mixture median thickness over the Earth for {name}]({prefix}{SPHERICAL}/world.png) |", "",
                  "Both maps use the vent longitude as their central meridian. Read the thickness key under each map.", "",
                  "</details>", ""])
    if corridor:
        index = lines.index("<summary>Compare the two whole-Earth maps</summary>")
        lines[index:index+7] = ["<summary>Compare the three whole-Earth maps</summary>", "",
            "| Uncertain fan direction | Spherical mixture | Nested corridor |", "|---|---|---|",
            f"| ![Directional whole-Earth median]({prefix}{DIRECTIONAL}/world.png) | "
            f"![Spherical whole-Earth median]({prefix}{SPHERICAL}/world.png) | "
            f"![Nested corridor whole-Earth median]({prefix}nested-isopachs/world.png) |", "",
            "Each map is centred on the vent longitude and has its own thickness key."]
    from ..models.directional.azimuth.primary import numerical_notes
    lines += [f"With the same observations and structure weights, fixing the fan directions gives a median-map volume "
              f"of **{centre['matched_control_map_km3']:,.1f} km³**.", ""]
    notes = numerical_notes(ensemble)
    if notes:
        lines += [" ".join(notes), ""]
    lines += ["<details>", "<summary>View the directional deposit extent and source area</summary>", "",
              f"![Principal uncertain-fan-direction median isopachs for {name}]({prefix}{DIRECTIONAL}/isopachs.png)",
              "", "</details>", ""]
    sensitivities = directional_sensitivities(home, ensemble, prefix)
    if sensitivities:
        lines[0] = ("## Four fits with interpreted archive limits" if regional["input"].get("interpreted_upper_limits")
                    else "## Four matched thickness-only fits")
        if nested:
            lines[0] = lines[0].replace("Four","Five")
        lines.extend(sensitivities)
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
