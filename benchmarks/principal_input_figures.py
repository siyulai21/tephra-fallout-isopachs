"""Render or verify the multiscale evidence figures for registered principal inputs."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import datasets
from benchmarks.principal_input_geography import PLACES, base, graticule, locator, places
from benchmarks.principal_input_sources import provenance_reader

ROOT = Path(__file__).resolve().parents[1]
PANEL_ROWS = 24
INK, MUTED, ACCENT = "#192e39", "#4d6570", "#912c55"
CMAP = "viridis_r"
REGION_COLOR = "#155963"
BOX_COLORS = ("#854315", "#72488c", "#126067", "#8b3650", "#385790", "#52611c")
# These frames separate the near-vent clusters from their surrounding survey.
# Recheck the hierarchy if an audit moves points or changes the panel partition.
NEAR_VENT = {
    "mazama": ("Crater Lake area", (-27., -202., 186., 125.)),
    "b-tm": ("Changbaishan–Tianchi area", (-47., -21., 71., 62.)),
    "ilopango-tbj": ("Lake Ilopango area", (-109., -48., 99., 80.)),
}
STYLES = {
    "Measured": ("o", ""),
    "Lower limit": ("^", "≥"),
    "Upper limit": ("v", "≤"),
    "Range": ("s", ""),
}
TABLE_COLUMNS = (
    "panel", "map_key", "input_row", "evidence_id", "site", "longitude", "latitude",
    "reported_kind", "reported_value_cm", "range_upper_cm", "fit_event", "source",
)


@dataclass(frozen=True)
class Panel:
    bounds: tuple[float, float, float, float]
    indices: np.ndarray


@dataclass(frozen=True)
class IndexBox:
    label: str
    bounds: tuple[float, float, float, float]
    color: str = ACCENT


def align_detail_panels(slug, xy, panels):
    """Separate rows that span a regional frame while retaining existing plate numbers."""
    if slug not in NEAR_VENT:
        return panels, []
    _, frame = NEAR_VENT[slug]
    revised = list(panels)
    splits = []

    def framed(indices, limit):
        bounds = padded_bounds(xy[indices], .115)
        bounds[:2] = np.maximum(bounds[:2], limit[:2])
        bounds[2:] = np.minimum(bounds[2:], limit[2:])
        return Panel(tuple(map(float, bounds)), indices)

    for i, panel in enumerate(panels):
        points = xy[panel.indices]
        inside = ((points >= frame[:2]) & (points <= frame[2:])).all(axis=1)
        if not inside.any() or inside.all():
            continue
        # Cut off south, north, west and east in turn. Once assigned,
        # a point outside two edges belongs to the first exterior piece.
        remaining = np.ones(len(points), dtype=bool)
        limits = np.array(panel.bounds)
        exterior = []
        for axis, upper in ((1, False), (1, True), (0, False), (0, True)):
            edge = frame[axis + 2] if upper else frame[axis]
            beyond = points[:, axis] > edge if upper else points[:, axis] < edge
            selected = remaining & beyond
            if selected.any():
                outer = limits.copy()
                outer[axis if upper else axis + 2] = edge
                exterior.append(framed(panel.indices[selected], outer))
            remaining &= ~beyond
            if upper:
                limits[axis + 2] = min(limits[axis + 2], edge)
            else:
                limits[axis] = max(limits[axis], edge)
        revised[i] = exterior[0]
        revised.extend(exterior[1:])
        revised.append(framed(panel.indices[inside], limits))
        splits.append({"original_detail": i + 1, "regional_detail": len(revised)})
    return revised, splits


def regional_view(slug, xy, panels):
    """Place every visible observation in one of the regional map's child boxes."""
    if slug not in NEAR_VENT:
        return None
    title, bounds = NEAR_VENT[slug]
    children = [i for i, panel in enumerate(panels, 1)
                if (np.array(panel.bounds[:2]) >= bounds[:2]).all()
                and (np.array(panel.bounds[2:]) <= bounds[2:]).all()]
    if not children:
        raise ValueError(f"No detail bounds fit inside the regional index for {slug}.")
    visible = ((xy >= bounds[:2]) & (xy <= bounds[2:])).all(axis=1)
    covered = np.concatenate([panels[i-1].indices for i in children])
    if not np.array_equal(np.sort(covered), np.flatnonzero(visible)):
        raise ValueError(f"Review the regional index frame for {slug} against the new panel layout.")
    return {"label": "A", "title": title, "file": "near-vent.png", "bounds_km": list(bounds),
            "detail_numbers": children, "observations": int(visible.sum())}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def audited_rows(record):
    data, meta = audited_data(record["path"])
    ledger, provenance = provenance_reader(record["slug"])
    with record["path"].open(newline="", encoding="utf-8-sig") as stream:
        raw = dict(enumerate(csv.DictReader(stream), 2))
    order = np.argsort(data.row_ids)
    source = data.reported_bounds()
    rows = []
    for i in order:
        number = int(data.row_ids[i])
        row = raw[number]
        if source.interval[i]:
            kind, value, upper = "Range", row["Lower_cm"], row["Upper_cm"]
        elif source.censored[i]:
            kind = "Lower limit" if source.lower_censored[i] else "Upper limit"
            value = row["Reported_lower_limit_cm"] if data.reported_minimum[i] else row["Limit_cm"]
            upper = ""
        else:
            kind, value, upper = "Measured", row["Thickness"], ""
        event = "Point" if not data.censored[i] else kind
        identity, source_reference, source_label = provenance(row, number)
        rows.append(dict(input_row=number, evidence_id=identity,
                         site=row.get("Site", ""), longitude=row["Lon"], latitude=row["Lat"],
                         reported_kind=kind, reported_value_cm=value, range_upper_cm=upper,
                         fit_event=event, source=source_reference, source_label=source_label))
    return data.xy[order], rows, meta, ledger


def padded_bounds(points, fraction):
    low, high = points.min(0), points.max(0)
    length = max((high - low).max(), .01)
    margin = length * fraction + np.maximum(0, length * .18 - (high - low)) / 2
    return np.r_[low - margin, high + margin]


def detail_panels(xy, overview):
    """Split at coordinate gaps and keep identical site positions on the same plate."""
    def partition(indices, clip, first=False):
        points = xy[indices]
        splits = []
        length = max(np.ptp(points, axis=0).max(), .01)
        for axis in range(2):
            order = np.argsort(points[:, axis], kind="stable")
            values = points[order, axis]
            for offset in np.flatnonzero(np.diff(values) > 0) + 1:
                balance = min(offset, len(order) - offset) / len(order)
                gap = (values[offset] - values[offset - 1]) / length
                score = gap * (.25 + balance) + .14 * balance
                if balance >= .27 or gap >= .38:
                    middle = (values[offset] + values[offset - 1]) / 2
                    splits.append((score, gap, axis, middle, order, offset))
        if splits:
            _, gap, axis, middle, order, offset = max(splits, key=lambda item: item[0])
            if first or len(indices) > PANEL_ROWS or (len(indices) >= 13 and gap >= .41):
                ordered = indices[order]
                left, right = clip.copy(), clip.copy()
                left[axis + 2], right[axis] = middle, middle
                return partition(ordered[:offset], left) + partition(ordered[offset:], right)
        padded = padded_bounds(points, .115)
        if np.ptp(points, axis=0).max() == 0:
            # A solitary point still needs enough geographic context to locate it.
            radius = .008 * (overview[2:] - overview[:2]).max()
            padded = np.r_[points[0] - radius, points[0] + radius]
        bounds = np.r_[np.maximum(padded[:2], clip[:2]), np.minimum(padded[2:], clip[2:])]
        return [Panel(tuple(map(float, bounds)), np.sort(indices))]

    return partition(np.arange(len(xy)), overview, first=True)


def panel_keys(panel, xy, rows):
    """Give coincident rows one marker and keep every row in the adjacent key."""
    _, groups = np.unique(xy[panel.indices], axis=0, return_inverse=True)
    keys = {}
    for i, group in zip(panel.indices, groups, strict=True):
        keys.setdefault(int(group), rows[i]["input_row"])
        yield i, keys[int(group)]


def axis_map(ax, bounds, vent, *, detail):
    base(ax, bounds, vent)
    ax.set(xlabel="Easting from vent (km)", ylabel="Northing from vent (km)")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(6))
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    if detail:
        graticule(ax, bounds, vent)
    else:
        ax.grid(color="#c8d8db", alpha=.65, linewidth=.55)

def draw_points(ax, xy, rows, norm, *, overview, regional_colors=None):
    points, groups = np.unique(xy, axis=0, return_inverse=True)
    shared = np.bincount(groups)[groups] > 1
    size = 31 if regional_colors else 21 if overview else 44
    edges = np.array([regional_colors[r["panel"]] if regional_colors else INK for r in rows])
    for kind, (marker, _) in STYLES.items():
        mask = np.array([r["reported_kind"] == kind for r in rows]) & ~shared
        if mask.any():
            values = [float(rows[i]["reported_value_cm"]) for i in np.flatnonzero(mask)]
            ax.scatter(*xy[mask].T, c=values, norm=norm, cmap=CMAP, marker=marker,
                       s=size, edgecolor=edges[mask], linewidth=1.04 if regional_colors else .62, zorder=5)
    if shared.any():
        ax.scatter(*points[np.unique(groups[shared])].T, marker="D", s=size + 5,
                   facecolor="white", edgecolor=INK, linewidth=.8, zorder=5)
    ax.scatter(0, 0, marker="*", s=143, facecolor="none", edgecolor=ACCENT, linewidth=1.25, zorder=6)


def label_positions(ax, anchors, labels, obstacles, *, colors=None, preferred_boxes=None):
    """Place labels in display space so marker and label spacing is scale independent."""
    transform = ax.transData
    ax.figure.canvas.draw()
    points = transform.transform(anchors)
    obstacles = transform.transform(obstacles)
    scale = ax.figure.dpi / 72
    lower = ax.bbox.min + 2 * scale
    upper = ax.bbox.max - 2 * scale
    angles = np.arange(16) * (np.pi / 8)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    offsets = np.vstack([radius * directions for radius in (13, 20, 28, 38, 52, 70, 92, 120)]) * scale
    used = []
    box_limits = None
    if preferred_boxes is not None:
        box_limits = [transform.transform(np.reshape(bounds, (2, 2)))
                      for bounds in preferred_boxes]
    for i, (point, label) in enumerate(zip(points, labels, strict=True)):
        extent = np.array([len(label) * 2.8 + 3.8, 7.0]) * scale
        color = colors[i] if colors is not None else INK

        def clear(center):
            lo, hi = center - extent, center + extent
            return (np.all(lo >= lower) and np.all(hi <= upper)
                    and not np.any(np.all(np.abs(obstacles - center) < extent + 3.5 * scale, axis=1))
                    and not any(np.all(lo < old_hi) and np.all(old_lo < hi) for old_lo, old_hi in used))

        candidates = point + offsets
        inner = None
        anchor = point
        if box_limits is not None:
            left, right = box_limits[i]
            inner_low, inner_high = left + extent + scale, right - extent - scale
            if np.all(inner_low <= inner_high):
                inner = next((p for p in (
                    inner_low + (inner_high - inner_low) * fraction
                    for fraction in ((x, y) for y in np.linspace(1., 0., 5)
                                     for x in (0., .25, .5, .75, 1.)))
                    if clear(p)), None)
            # Try immediately next to every edge before reaching out into the map.
            x = np.linspace(left[0], right[0], 7)
            y = np.linspace(left[1], right[1], 7)
            gap = extent + 1.2 * scale
            near_edges = np.vstack((np.c_[x, np.full(7, left[1] - gap[1])],
                                    np.c_[x, np.full(7, right[1] + gap[1])],
                                    np.c_[np.full(7, left[0] - gap[0]), y],
                                    np.c_[np.full(7, right[0] + gap[0]), y]))
            candidates = np.vstack((near_edges, candidates))
            distances = np.linalg.norm(candidates - np.clip(candidates, left, right), axis=1)
            candidates = candidates[np.argsort(distances, kind="stable")]
            # A badge outside its own box should stay outside every other box.
            candidates = [p for p in candidates if not any(
                np.all(p + extent > lo) and np.all(p - extent < hi)
                for j, (lo, hi) in enumerate(box_limits) if j != i)]
        center = inner if inner is not None else next((p for p in candidates if clear(p)), None)
        if center is None:
            raise ValueError(f"Could not place map label {label}. Review the map extent.")
        used.append((center - extent, center + extent))
        endpoint = transform.inverted().transform(center)
        if inner is None:
            if box_limits is not None:
                anchor = np.clip(center, *box_limits[i])
            start = transform.inverted().transform(anchor)
            ax.plot(*np.stack((start, endpoint)).T, color=color, linewidth=.55, alpha=.75, zorder=6)
        ax.text(*endpoint, label, ha="center", va="center", color=color, fontsize=8,
                fontweight="bold", zorder=7, bbox=dict(boxstyle="round,pad=.18",
                facecolor="white", edgecolor=color, linewidth=.55))


def regional_boxes(panels, numbers):
    boxes = []
    for j, number in enumerate(numbers):
        bounds = panels[number - 1].bounds
        low, high = np.reshape(bounds, (2, 2))
        touching_colors = set()
        for box in boxes:
            other_low, other_high = np.reshape(box.bounds, (2, 2))
            # At regional scale even a gap of a kilometre can look like a shared edge.
            if (np.maximum(low, other_low) <= np.minimum(high, other_high) + 1.).all():
                touching_colors.add(box.color)
        preferred = j % len(BOX_COLORS)
        palette = BOX_COLORS[preferred:] + BOX_COLORS[:preferred]
        color = next((color for color in palette if color not in touching_colors), palette[0])
        boxes.append(IndexBox(f"{number:02}", bounds, color))
    return boxes


def draw_index_boxes(ax, boxes, obstacles, *, regional):
    corners = []
    for box in boxes:
        x0, y0, x1, y1 = box.bounds
        lettered = box.label.isalpha()
        if regional or lettered:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=box.color,
                                   edgecolor="none", alpha=.12, zorder=3.3))
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, zorder=4,
                               edgecolor="white", linewidth=2.7 if lettered else 2.4))
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, zorder=4,
                               edgecolor=box.color, linewidth=1.8 if lettered else 1.1 if regional else .95,
                               linestyle="-" if regional or lettered else (0, (4, 2))))
        corners.append((x0, y1))
    label_positions(ax, corners, [box.label for box in boxes], obstacles,
                    colors=[box.color for box in boxes],
                    preferred_boxes=[box.bounds for box in boxes] if regional else None)


def reading(row):
    first = f'{float(row["reported_value_cm"]):.5g}'
    if row["range_upper_cm"]:
        upper = f'{float(row["range_upper_cm"]):.5g}'
        # Do not display a narrow range as two equal rounded values.
        if first == upper:
            first, upper = row["reported_value_cm"], row["range_upper_cm"]
        return f"{first}–{upper}"
    return STYLES[row["reported_kind"]][1] + first


def table(fig, rows):
    ax = fig.add_axes([.525, .171, .451, .695])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    columns = [(.018, "Row", "left"), (.102, "EvidenceID", "left"),
               (.380, "Value (cm)", "right"), (.405, "Fit event", "left"), (.575, "Source", "left")]
    for x, heading, align in columns:
        ax.text(x, 1, heading, va="top", ha=align, weight="bold", color=INK, fontsize=9)
    step = min(.061, .944 / len(rows))
    for j, row in enumerate(rows):
        top = .943 - j * step
        if j % 2 == 0:
            ax.add_patch(Rectangle((0, top - step), 1, step, facecolor="#edf2f2", edgecolor="none"))
        key = str(row["map_key"])
        if row["map_key"] != row["input_row"]:
            key += f' / {row["input_row"]}'
        citation = "\n".join(textwrap.wrap(row["source_label"] or row["source"], width=45,
                                              max_lines=2, placeholder="…"))
        labels = (key, row["evidence_id"] or "—", reading(row), row["fit_event"], citation)
        for (x, heading, align), label in zip(columns, labels, strict=True):
            ax.text(x, top - step / 2, label, va="center", ha=align, color=INK,
                    fontsize=8.2 if heading == "Source" else 9, linespacing=1.07)
    fig.text(.532, .131, "Full citations and coordinates are in sites.csv.  —  Evidence ID unavailable.",
             fontsize=8.5, color=MUTED)

def figure(path, title, subtitle, bounds, vent, xy, rows, norm, kinds, *, boxes=None,
           coincident=False, regional=False):
    overview = boxes is not None
    size = (10.7, 11.2) if regional and bounds[3]-bounds[1] > bounds[2]-bounds[0] else (11.6, 8)
    fig = plt.figure(figsize=size if overview else (15, 9.5), facecolor="white", dpi=160)
    try:
        fig.text(.073, .946, title, fontsize=21, weight="bold", color=INK)
        fig.text(.074, .902, subtitle, fontsize=11, color=MUTED)
        frame = [.095, .232, .81, .622] if overview else [.073, .323, .423, .544]
        if regional:
            frame = [.097, .217, .807, .651]
        ax = fig.add_axes(frame)
        axis_map(ax, bounds, vent, detail=regional or not overview)
        regional_colors = {box.label: box.color for box in boxes} if regional else None
        draw_points(ax, xy, rows, norm, overview=overview, regional_colors=regional_colors)
        if overview:
            obstacles = np.vstack((xy, [0, 0]))
            draw_index_boxes(ax, boxes, obstacles, regional=regional)
            if regional:
                places(ax, bounds, vent, obstacles=obstacles)
                fig.text(.096, .162, "Marker outlines match the numbered detail boxes.",
                         fontsize=9, color=MUTED)
        else:
            _, first = np.unique([r["map_key"] for r in rows], return_index=True)
            label_positions(ax, xy[first], [str(rows[i]["map_key"]) for i in first],
                            np.vstack((xy, [0, 0])))
            places(ax, bounds, vent, obstacles=np.vstack((xy, [0, 0])))
            locator(fig, bounds, vent)
            table(fig, rows)
        handles = [Line2D([], [], marker=m, markersize=7, ls="", color=INK,
                          markerfacecolor="white", label=k)
                   for k, (m, _) in STYLES.items() if k in kinds]
        if coincident:
            handles.append(Line2D([], [], marker="D", ls="", color=INK, markerfacecolor="white",
                                  label="Shared coordinates"))
        handles.append(Line2D([], [], marker="*", markersize=10, color=ACCENT, markerfacecolor="none", ls="", label="Vent"))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
                           cax=fig.add_axes([.096, .101, .48, .018] if overview else [.075, .045, .38, .013]), orientation="horizontal",
                           ticks=colour_ticks(norm))
        bar.ax.xaxis.set_minor_locator(NullLocator())
        bar.ax.set_xticklabels([f"{x:,.0f}" if x >= 1 and float(x).is_integer() else f"{x:.3g}" for x in bar.get_ticks()])
        bar.ax.tick_params(labelsize=8, colors=MUTED, length=3)
        bar.outline.set_linewidth(.45)
        scale_label = "Marker colour · reported thickness or limit (cm)"
        if "Range" in kinds:
            scale_label += " · lower endpoint for ranges"
        text_x, text_y = (.096, .133) if overview else (.075, .069)
        fig.text(text_x, text_y, scale_label, fontsize=9, color=MUTED)
        legend_position = (.084, .014) if overview else (.52, .018)
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=legend_position, ncols=len(handles),
                   frameon=False, fontsize=9, labelcolor=INK, handletextpad=.27)
        fig.savefig(path, dpi=160, facecolor="white")
    finally:
        plt.close(fig)


def colour_ticks(norm):
    decades = 10. ** np.arange(np.ceil(np.log10(norm.vmin)), np.floor(np.log10(norm.vmax)) + 1)
    if len(decades) < 3:
        return np.geomspace(norm.vmin, norm.vmax, 5)
    return decades


def packet(record):
    xy, rows, meta, ledger = audited_rows(record)
    bounds = padded_bounds(np.vstack((xy, [0, 0])), .09)
    panels, splits = align_detail_panels(record["slug"], xy, detail_panels(xy, bounds))
    assigned = np.zeros(len(xy), dtype=int)
    for number, panel in enumerate(panels, 1):
        for i, key in panel_keys(panel, xy, rows):
            rows[i].update(panel=f"{number:02}", map_key=key)
            assigned[i] += 1
        points = xy[panel.indices]
        if not ((points >= panel.bounds[:2]) & (points <= panel.bounds[2:])).all():
            raise ValueError("A detail plate excludes an assigned observation.")
    if not (assigned == 1).all():
        raise ValueError("Each accepted row must appear on one detail plate.")
    counts = Counter(r["reported_kind"] for r in rows)
    values = [float(r["reported_value_cm"]) for r in rows]
    values.extend(float(r["range_upper_cm"]) for r in rows if r["range_upper_cm"])
    low, high = min(values), max(values)
    if low == high:
        low, high = low / 1.1, high * 1.1
    manifest = {
        "input": f'data/{record["file"]}',
        "input_sha256": meta["input_sha256"],
        "projection": meta["projection"],
        "vent_lonlat": meta["vent_lonlat"],
        "reported_counts": dict(sorted(counts.items())),
        "n_rows": len(rows),
        "pseudo_rows_omitted": meta["ignored_pseudo_rows"],
        "overview_bounds_km": bounds.tolist(),
        "colour_limits_cm": [low, high],
        "basemap_places_sha256": digest(PLACES),
        "observations_without_evidence_id": sum(not row["evidence_id"] for row in rows),
        "details": [{"file": f"detail-{number:02}.png", "bounds_km": list(panel.bounds),
                     "input_rows": [rows[i]["input_row"] for i in panel.indices]}
                    for number, panel in enumerate(panels, 1)],
    }
    region = regional_view(record["slug"], xy, panels)
    if region:
        if splits:
            region["split_details"] = splits
        manifest["regional_index"] = region
    if ledger:
        manifest["row_sources"] = ledger.relative_to(ROOT).as_posix()
        manifest["row_sources_sha256"] = digest(ledger)
    if "reported_source" in meta:
        manifest["reported_input"] = meta["reported_source"]
        manifest["reported_input_sha256"] = meta["reported_source_sha256"]
    if "audit" in meta:
        manifest["audit"] = f'data/{meta["audit"]["file"]}'
        manifest["audit_sha256"] = meta["audit"]["sha256"]
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=TABLE_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return xy, rows, panels, manifest, index.getvalue()


def case_page(record, manifest, directory):
    title = record["name"]
    link = os.path.relpath(record["path"], directory)
    source_links = [f"[Accepted input CSV]({link})"]
    for field, label in (("reported_input", "Reported CSV"), ("audit", "Audit provenance"), ("row_sources", "Source ledger")):
        if field in manifest:
            source_links.append(f"[{label}]({os.path.relpath(ROOT / manifest[field], directory)})")
    if record["slug"] == "b-tm":
        bibliography = ROOT / "data/audits/B-Tm/two-phase/sources.json"
        source_links.append(f"[Audit citation keys]({os.path.relpath(bibliography, directory)})")
    source_links += ["[Site index with coordinates and full input precision](sites.csv)",
                     "[Input hashes and panel bounds](manifest.json)"]
    region = manifest.get("regional_index")
    if region:
        source_links.insert(0, "[Go to the near-vent map](#near-vent-detail-index)")
    box_description = ("Box A on the overview marks the area shown in the near-vent detail index. "
                       "Other numbered boxes lead directly to detail plates. "
                       if region else "The dashed boxes show the detail extents. ")
    lines = [f"# {title} · principal input observations", "",
             f'All {manifest["n_rows"]} accepted rows have a symbol at their registered coordinates on the overview '
             "and a labelled reading on one numbered detail plate. " + box_description +
             "A star marks the vent. Positions use a vent-centred equal-area grid in kilometres. The basemap uses Natural Earth land and coastlines.", "",
             " · ".join(source_links), "",
             "The table shows the evidence ID and source citation alongside the reported or interpreted reading. "
             "A dash marks an unavailable evidence ID. Complete citations and unrounded values are in the site index. "
             "A reported lower limit appears as ≥ on these evidence figures and as a point at that value "
             "in the fit. Upper limits and ranges retain their respective events. Marker colour represents "
             "the displayed thickness or limit, using the lower endpoint for ranges. Where rows share "
             "coordinates, a single diamond uses the first CSV row as its map key and the adjacent table lists each row.", "",
             "![Overview of accepted observation locations and numbered detail boxes](overview.png)", ""]
    if region:
        links = ", ".join(f'[{n:02}](detail-{n:02}.png)' for n in region["detail_numbers"])
        lines += ["## Near-vent detail index", "",
                  f'**Box A · {region["title"]}** contains {region["observations"]} accepted observations. '
                  f'Match a numbered box to the links below for its site locations and readings. {links}', "",
                  f'![Enlarged box A with numbered boundaries for the near-vent detail plates]({region["file"]})', ""]
        for split in region.get("split_details", []):
            prior, local = split["original_detail"], split["regional_detail"]
            lines += [f'Sites previously grouped in [Detail {prior:02}](detail-{prior:02}.png) '
                      f'now have a separate [Detail {local:02}](detail-{local:02}.png) inside Box A.', ""]
        lines += ["## All detail plates", ""]
    lines += ["| Detail plate | Input rows | Easting × northing span |", "|---|---:|---:|"]
    for number, detail in enumerate(manifest["details"], 1):
        x0, y0, x1, y1 = detail["bounds_km"]
        group = " · A" if region and number in region["detail_numbers"] else ""
        lines.append(f'| [{number:02}]({detail["file"]}){group} | {len(detail["input_rows"])} | '
                     f'{x1 - x0:.3g} × {y1 - y0:.3g} km |')
    return "\n".join(lines) + "\n"


def run_case(record, directory, *, verify):
    xy, rows, panels, manifest, index = packet(record)
    region = manifest.get("regional_index")
    images = ["overview.png", *(detail["file"] for detail in manifest["details"])]
    if region:
        images.append(region["file"])
    if verify:
        expected = json.loads((directory / "manifest.json").read_text())
        # PROJ roundoff can differ across platforms. Compare coordinates before checking
        # the published display extents and their rounded captions.
        pairs = [(manifest, expected, "overview_bounds_km")]
        pairs += [(actual, saved, "bounds_km") for actual, saved in
                  zip(manifest["details"], expected["details"], strict=True)]
        for actual, saved, field in pairs:
            np.testing.assert_allclose(actual[field], saved[field], rtol=0, atol=1e-8)
            actual[field] = saved[field]
        if (directory / "sites.csv").read_bytes() != index.encode():
            raise ValueError(f"Published site index is stale for {record['slug']}.")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        title = record["name"]
        vent = tuple(manifest["vent_lonlat"])
        norm = LogNorm(*manifest["colour_limits_cm"])
        kinds = manifest["reported_counts"]
        coincident = len(np.unique(xy, axis=0)) != len(xy)
        numbers = set(region["detail_numbers"]) if region else set()
        boxes = [IndexBox(f"{n:02}", panel.bounds) for n, panel in enumerate(panels, 1)
                 if n not in numbers]
        subtitle = f"{len(rows)} accepted observations  |  dashed boxes mark numbered detail plates"
        if region:
            boxes.append(IndexBox(region["label"], region["bounds_km"], REGION_COLOR))
            subtitle = f"{len(rows)} accepted observations  |  box A marks the near-vent detail index"
        figure(directory / "overview.png", title, subtitle,
               manifest["overview_bounds_km"], vent, xy, rows, norm, kinds, boxes=boxes,
               coincident=coincident)
        if region:
            children = region["detail_numbers"]
            subset = np.sort(np.concatenate([panels[n-1].indices for n in children]))
            detail_boxes = regional_boxes(panels, children)
            figure(directory / region["file"], f'{title} · A · {region["title"]}',
                   f'{len(subset)} accepted observations  |  numbers match the detail plates below this map',
                   region["bounds_km"], vent, xy[subset], [rows[i] for i in subset], norm, kinds,
                   boxes=detail_boxes, coincident=coincident, regional=True)
        for number, (panel, detail) in enumerate(zip(panels, manifest["details"], strict=True), 1):
            group = [rows[i] for i in panel.indices]
            figure(directory / detail["file"], title,
                   f"Detail {number:02} of {len(panels):02}  |  {len(group)} accepted rows",
                   panel.bounds, vent, xy[panel.indices], group, norm, kinds, coincident=coincident)
        (directory / "sites.csv").write_bytes(index.encode())
        (directory / "README.md").write_text(case_page(record, manifest, directory), encoding="utf-8")
    manifest["files_sha256"] = {name: digest(directory / name) for name in (*images, "sites.csv", "README.md")}
    if verify:
        if (directory / "README.md").read_text(encoding="utf-8") != case_page(record, manifest, directory) or expected != manifest:
            raise ValueError(f"Published observation figures are stale for {record['slug']}.")
    else:
        old_maps = set(directory.glob("detail-*.png")) | {directory / "near-vent.png"}
        for old in old_maps - {directory / name for name in images}:
            old.unlink(missing_ok=True)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(record["slug"], len(rows), "observations,", len(panels), "detail plates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="*", help="Registered eruption slugs; defaults to all")
    parser.add_argument("--verify", action="store_true", help="Check input, panel coverage and published file hashes")
    args = parser.parse_args()
    records = {rec["slug"]: rec for rec in datasets()}
    if unknown := set(args.case) - records.keys():
        parser.error(f"Unknown dataset: {', '.join(sorted(unknown))}")
    for slug in args.case or records:
        run_case(records[slug], ROOT / "eruptions" / slug / "observations", verify=args.verify)


if __name__ == "__main__":
    main()
