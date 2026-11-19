"""Global and local views of saved posterior fields; display does not set mass."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize

from src.Laea import make_laea_transform
from src.probabilistic_isopach.global_volume import EarthDomain
from src.probabilistic_isopach.figures import STYLE
from .maps import (ASSET, INK, TEAL, THICKNESS_LEVELS, frame_bounds, grid_axes, global_frame,
                   panel_layout, setup_map, observations, observation_legend,
                   coast_rings, level_labels, contour_levels)


class PosteriorView:
    """Deterministic, chain-spanning subset used for plotting only."""
    def __init__(self, samples, latitude, geometry=None, maximum=512):
        from ..models.mass_mixture.inference import flatten
        from ..models.directional.inference import flat
        self.g = geometry
        self.domain = EarthDomain.wgs84(latitude)
        self.latitude = latitude
        raw = flat(samples) if geometry is not None else flatten(samples)
        key = "source" if geometry is not None else "log_volume"
        indices = np.unique(np.linspace(0, len(raw[key])-1, min(maximum, len(raw[key]))).astype(int))
        self.samples = {k: v[indices] for k, v in raw.items()}
        self.n = len(indices)
        self.indices = indices

    def evaluate(self, xy):
        from ..models.mass_mixture.geometry import log_field, to_sphere
        from ..models.directional.inference import field
        xy = np.asarray(xy)
        out = np.full((3, len(xy)), np.nan)
        physical = np.flatnonzero(self.domain.rho(xy) < self.domain.rho_max)
        for start in range(0, len(physical), 256):
            ids = physical[start:start+256]
            values = (field(self.samples, self.g, xy[ids]) if self.g is not None
                      else log_field(self.samples, to_sphere(xy[ids], self.latitude)))
            q = np.quantile(values, [.025, .5, .975], axis=0)
            if self.g is not None:
                # Preserve the directional report's original mean-log surface.
                q[1] = values.mean(0)
            out[:, ids] = q/np.log(10.)
        return out

    def central(self, xy):
        return self.evaluate(xy)[1]

    @property
    def statistic(self):
        return "Geometric-mean thickness" if self.g is not None else "Posterior median thickness"


def grid(view, bounds, longest=321):
    x, y = grid_axes(bounds, longest=longest)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    return x, y, view.evaluate(xy).reshape(3, len(y), len(x))


def footprint(ax, mesh, color=TEAL, lw=1.):
    for polygon in getattr(mesh.geometry, "geoms", [mesh.geometry]):
        if not polygon.is_empty:
            for ring in [polygon.exterior, *polygon.interiors]:
                ax.plot(*ring.xy, color=color, lw=lw, ls="--", zorder=5)


def field_panel(ax, x, y, z, bounds, data, meta, norm, mesh=None):
    image = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis", norm=norm,
                          rasterized=True, zorder=1)
    levels = contour_levels(z)
    if len(levels):
        contour = ax.contour(x, y, z, levels=levels, colors="white", linewidths=.75,
                             linestyles="solid", zorder=4)
        ax.clabel(contour, fmt=level_labels, fontsize=8, inline_spacing=5)
    if mesh is not None:
        footprint(ax, mesh, color="#9BE3DB", lw=.9)
    setup_map(ax, bounds, meta["vent_lonlat"])
    observations(ax, data)
    return image


def thickness_bar(fig, image, axes, norm):
    bar = fig.colorbar(image, ax=axes, location="bottom", shrink=.8, pad=.06, aspect=44,
                      label="Thickness (cm, logarithmic colour scale)")
    ticks = np.log10(THICKNESS_LEVELS)
    ticks = ticks[(ticks >= norm.vmin) & (ticks <= norm.vmax)]
    if len(ticks) > 9:
        ticks = ticks[::2]
    bar.set_ticks(ticks, labels=[level_labels(t) for t in ticks])


def save(fig, output, name):
    fig.savefig(Path(output)/name, dpi=190, bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def world_map(output, name, data, meta, view, norm, *, show_contours=True):
    """Mollweide overview: the left and right edges are the same meridian."""
    longitude = np.linspace(-180., 180., 361)
    latitude = np.linspace(-89.95, 89.95, 181)
    lon, lat = np.meshgrid(longitude, latitude)
    geographic_lon = (lon+meta["vent_lonlat"][0]+180.) % 360.-180.
    forward, inverse, _ = make_laea_transform(*meta["vent_lonlat"])
    xy = np.column_stack(forward.transform(geographic_lon.ravel(), lat.ravel()))
    z = view.central(xy).reshape(lon.shape)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12.5, 7.1), subplot_kw={"projection":"mollweide"}, layout="constrained")
        image = ax.pcolormesh(np.deg2rad(longitude), np.deg2rad(latitude), z, norm=norm,
                              cmap="viridis", shading="auto", rasterized=True)
        levels = contour_levels(z)
        levels = levels[::2]  # 0.1, 1, 10, 100 ...; local figures show intermediate levels.
        if show_contours and len(levels):
            lines = ax.contour(np.deg2rad(longitude), np.deg2rad(latitude), z,
                               levels=levels, colors="white", linewidths=.6)
            ax.clabel(lines, fmt=level_labels, fontsize=7)
        coast = []
        for ring in coast_rings():
            relative = (ring[:, 0]-meta["vent_lonlat"][0]+180) % 360-180
            points = np.deg2rad(np.column_stack((relative, ring[:, 1])))
            split = np.flatnonzero(abs(np.diff(relative)) > 180)+1
            coast.extend(p for p in np.split(points, split) if len(p) > 1)
        ax.add_collection(LineCollection(coast, colors="#A5B7B4", linewidths=.35,
                                         zorder=3, rasterized=True), autolim=False)
        dl, dt = inverse.transform(*data.xy.T)
        dl = (np.asarray(dl)-meta["vent_lonlat"][0]+180) % 360-180
        ax.scatter(np.deg2rad(dl), np.deg2rad(dt), s=6, facecolors="white", edgecolors=INK, lw=.25, zorder=5)
        ax.scatter(0, np.deg2rad(meta["vent_lonlat"][1]), marker="*", s=90,
                   c="#F18B60", edgecolors="white", lw=.6, zorder=6)
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=8)
        ax.set_xticklabels([""]*len(ax.get_xticks()))
        legend = "contours in cm" if show_contours else "thickness in cm"
        ax.set_title(f"{name} · whole-Earth context\n{view.statistic} · {legend}", fontsize=14, pad=18)
        thickness_bar(fig, image, [ax], norm)
        fig.supxlabel("Mollweide overview centred on the source longitude · map edges join · unsampled colour is conditional extrapolation", fontsize=9)
        save(fig, output, "world.png")
    return {"projection":"Mollweide, source longitude at centre", "grid_shape":list(z.shape),
            "edge_note":"Longitude edges join; polar display rows stop at ±89.95°. This display raster is not used to integrate volume."}


def global_maps(output, name, data, meta, samples, report, g=None, arcs=None):
    from ..support import SupportMesh
    output = Path(output)
    view = PosteriorView(samples, meta["vent_lonlat"][1], geometry=g)
    bounds, audit = global_frame(data.xy, view.latitude, view.central)
    local_bounds = frame_bounds(data.xy)
    x, y, q = grid(view, bounds)
    lx, ly, lq = grid(view, local_bounds)
    high = max(float(np.quantile(data.log_value[~data.censored]/np.log(10.), .98)),
               float(np.nanquantile(lq[1], .995)))
    norm = Normalize(-1., max(high, 1.))
    mesh = SupportMesh(data) if g is None else g.mesh
    numerical = (report["fits"]["baseline"]["diagnostics"]["passed"] if g is None else report["passed"])
    warning = "" if numerical else " · SAMPLING WARNING"
    edge_note = ("Expanded view contains the displayed 0.1 cm contour in the finite-grid edge check."
                 if audit["lowest_contour_within_chart"] else
                 "Some contours reach the projection rim; follow them in the whole-Earth companion.")
    with plt.rc_context(STYLE):
        if g is None:
            shape, size = panel_layout(bounds)
            fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", squeeze=False)
            left, right = axes.flat
            image = field_panel(left, x, y, q[1], bounds, data, meta, norm, mesh)
            left.set_title("Whole-deposit field\nPosterior median; contours in cm", fontsize=11)
            uncertainty = right.pcolormesh(x, y, q[2]-q[0], shading="auto", cmap="magma", vmin=0, rasterized=True)
            setup_map(right, bounds, meta["vent_lonlat"])
            observations(right, data)
            footprint(right, mesh, color="#9BE3DB")
            right.set_title("Latent thickness uncertainty\nObservation scatter is additional", fontsize=11)
            thickness_bar(fig, image, [left], norm)
            fig.colorbar(uncertainty, ax=right, location="bottom", shrink=.8, pad=.06, aspect=44,
                         label="Orders of magnitude: upper / lower 95% thickness")
            fig.suptitle(f"{name} · spherical-mixture global reconstruction{warning}\n{edge_note}", fontsize=13)
            observation_legend(fig, data)
            save(fig, output, "field.png")
        else:
            # Different frames are intentional: fragments need their local
            # context while continuation needs room to show its closing levels.
            ratios = [(b[2]-b[0])/(b[3]-b[1]) for b in (local_bounds, bounds)]
            if min(ratios) > 1.6:
                shape, size = (2, 1), (12., 10.6/ratios[0]+10.6/ratios[1]+2.4)
                proportions = {}
            else:
                shape, size = (1, 2), (13.2, max(5.8, 11.7/sum(ratios)+2.2))
                proportions = {"width_ratios":ratios}
            fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", squeeze=False,
                                     gridspec_kw=proportions)
            left, right = axes.flat
            left.add_collection(PolyCollection(mesh.triangles, facecolors="#D3E9E5", edgecolors="none", zorder=1))
            selected = arcs[arcs.displayed.astype(bool)]
            if len(selected):
                segments = selected[["x1_km","y1_km","x2_km","y2_km"]].to_numpy().reshape(-1, 2, 2)
                lines = LineCollection(segments, cmap="viridis", norm=norm, linewidths=2, zorder=4)
                lines.set_array(np.log10(selected.level_cm))
                left.add_collection(lines, autolim=False)
            setup_map(left, local_bounds, meta["vent_lonlat"])
            observations(left, data)
            left.set_title("Measured isopach fragments\nComplete observation extent; gaps stay open", fontsize=11)
            image = field_panel(right, x, y, q[1], bounds, data, meta, norm, mesh)
            right.set_title("Wider conditional continuation\nGeometric-mean thickness; contours in cm", fontsize=11)
            thickness_bar(fig, image, list(axes.flat), norm)
            observation_legend(fig, data)
            # Reserve a separate header above the constrained axes. Fixed-aspect
            # panels with unequal width ratios otherwise collide with it.
            fig.suptitle(f"{name} · directional global reconstruction{warning}\nPanels use different extents. {edge_note}",
                         fontsize=12, y=1.015, va="bottom")
            save(fig, output, "isopachs.png")
        # Same complete local extent for both global families: readers can see
        # the observations even when the long tail requires a continental view.
        shape, size = panel_layout(local_bounds)
        fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", squeeze=False)
        left, right = axes.flat
        image = field_panel(left, lx, ly, lq[1], local_bounds, data, meta, norm, mesh)
        left.set_title(f"Near-observation field\n{view.statistic}", fontsize=11)
        uncertainty = right.pcolormesh(lx, ly, lq[2]-lq[0], shading="auto", cmap="magma", vmin=0, rasterized=True)
        setup_map(right, local_bounds, meta["vent_lonlat"])
        observations(right, data)
        footprint(right, mesh, color="#9BE3DB")
        right.set_title("Local latent uncertainty\nCyan outlines: sampling footprint", fontsize=11)
        thickness_bar(fig, image, [left], norm)
        fig.colorbar(uncertainty, ax=right, location="bottom", shrink=.8, pad=.06, aspect=44,
                     label="Orders of magnitude: upper / lower 95% thickness")
        fig.suptitle(f"{name} · local detail{warning}\nA detail window is not the global integration boundary; contours can continue outside it.", fontsize=13)
        observation_legend(fig, data)
        save(fig, output, "local.png")
    world = world_map(output, name, data, meta, view, norm)
    provenance = {"protocol":"report-cartography-v1", "input_sha256":meta["input_sha256"],
        "posterior_sha256":sha256((output/"posterior.npz").read_bytes()).hexdigest() if (output/"posterior.npz").exists() else None,
        "coastline_sha256":sha256(ASSET.read_bytes()).hexdigest(), "coastline_scale":"1:10 million",
        "central_statistic":view.statistic, "posterior_fields":view.n,
        "posterior_flat_indices_sha256":sha256(view.indices.astype('<i8').tobytes()).hexdigest(),
        "global_bounds_km":bounds.tolist(), "local_bounds_km":local_bounds.tolist(),
        "global_grid_shape":list(q.shape[1:]), "local_grid_shape":list(lq.shape[1:]),
        "extent_audit":audit, "whole_earth_view":world,
        "volume_calculation_changed":False,
        "note":"Finite plotting grids summarize saved fields. Thresholds and map windows do not enter inference or volume."}
    (output/"map-display.json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    if g is not None:
        index = int(np.nanargmax(q[1]))
        xx, yy = np.meshgrid(x, y)
        peak = np.array([xx.ravel()[index], yy.ravel()[index]])
        from ..models.directional.geometry import polar
        from shapely.geometry import Point
        triangle = g.locate([peak])[0][0]
        return {"grid_maximum_chord_distance_km":float(polar(peak, g.axis)[0]),
            "grid_maximum_region":("retained measurement triangles" if g.mesh.geometry.covers(Point(peak)) else "transition" if triangle >= 0 else "background continuation"),
            "map_window_xy_km":bounds.tolist(), "posterior_fields_for_map":view.n,
            "source_geometric_mean_cm":float(10**view.central([[0., 0.]])[0]),
            "grid_maximum_cm":float(10**q[1].ravel()[index]), "grid_maximum_xy_km":peak.tolist(),
            "display_audit":"map-display.json"}
    return provenance
