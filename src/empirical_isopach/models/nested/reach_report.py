"""Case-level inference, maps, geographic validation and shared contour screen."""
from pathlib import Path
import csv
import gzip
import io
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.special import ndtr
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.probabilistic_isopach.global_volume import EarthDomain
from ...contour_support import audit_contours
from ...core.audit import audited_data
from ...core.catalog import dataset_record
from ...core.geometry import survey_bounds
from ...support import SupportMesh
from ...visualization.global_maps import thickness_bar, world_map
from ...visualization.maps import frame_bounds, grid_axes, observations, observation_legend, setup_map
from ..directional.ensemble.members import splits
from .continuation import from_parent, load_packet
from .diagnostics import check_bounds
from .geometry import DIAMETER
from .profile import parameters
from .reach import LAWS, PRIOR, PROTOCOL, PROBS
from .reach_fit import conditioned_fit, signature
from .reach_figure import compass_map
from .reach_integrals import LEVELS, SECTORS, checked, nodes
from .report import MapView, archive_predictions
from .study import digest, infer_dataset, write_json


def bracketed_regions(data, mesh, model, levels):
    """Retain the exact-site brackets and posterior crossing probability at each level."""
    sites = [data.log_value[(data.xy == xy).all(axis=1) & ~data.censored] for xy in mesh.xy]
    low = np.asarray([min(v) for v in sites])
    high = np.asarray([max(v) for v in sites])
    idx = mesh.triangulation.simplices
    vals = model.reduced(1024).field(mesh.xy)[:, idx]
    lower, upper = vals.min(axis=2), vals.max(axis=2)
    regions, records = {}, []
    for level in levels:
        t = level * np.log(10.)
        pair = (high[idx] < t).any(axis=1) & (low[idx] > t).any(axis=1)
        prob = np.mean((lower < t) & (upper > t), axis=0)
        use = mesh.accepted & pair & (prob >= .9)
        regions[level] = unary_union([Polygon(mesh.xy[v]) for v in idx[use]])
        for j in range(len(idx)):
            records.append((10.**level, j, *idx[j], int(mesh.accepted[j]), int(pair[j]), float(prob[j]), int(use[j])))
    return regions, records


def line_parts(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == "LineString":
        return [np.asarray(shape.coords)]
    if hasattr(shape, "geoms"):
        return [line for part in shape.geoms for line in line_parts(part)]
    return []


def contour_segments(ax, x, y, z, data, mesh, deposit):
    rows = []
    levels = [v for v in (-1, 0, 1, 2, 3) if np.nanmin(z) < v < np.nanmax(z)]
    if levels:
        cs = ax.contour(x, y, z, levels=levels, colors="white", linewidths=1.05, linestyles="dashed")
        regions, rows = bracketed_regions(data, mesh, deposit, levels)
        for level, segments in zip(levels, cs.allsegs, strict=True):
            fragments = [piece for line in segments if len(line) >= 2 for piece in
                         line_parts(LineString(line).intersection(regions[level]))]
            ax.add_collection(LineCollection(fragments, colors="white", linewidths=1.9, zorder=5))
        ax.clabel(cs, fmt=lambda v:f"{10**v:g}", fontsize=8)
    return rows


def principal_map(data, meta, name, out, model, report):
    bounds = frame_bounds(data.xy)
    x, y = grid_axes(bounds, longest=181)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    shown = model.reduced(512)
    z = MapView(shown).central(xy).reshape(xx.shape)
    norm = Normalize(-2., max(2., np.ceil(np.nanmax(z))))
    fig, ax = plt.subplots(figsize=(8.3, 6.5), layout="constrained")
    image = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis", norm=norm, rasterized=True)
    contour_segments(ax, x, y, z, data, SupportMesh(data), model)
    setup_map(ax, bounds, meta["vent_lonlat"])
    observations(ax, data, size=20, meta=meta)
    lo, mid, hi = report["volume_km3"]
    warning = " · numerical warning" if not all(report["numerics"].values()) else ""
    fig.suptitle(f"{name} · Bayesian nested corridor", fontsize=16, fontweight="bold")
    ax.set_title(f"Median isopachs (cm)   |   Bulk volume {mid:,.1f} km³ [{lo:,.1f}–{hi:,.1f}]{warning}", fontsize=10)
    thickness_bar(fig, image, ax, norm)
    observation_legend(fig, data, [Line2D([], [], color="#274552", lw=2, label="Locally bracketed"),
                                   Line2D([], [], color="#274552", ls="--", label="Continuation")], meta=meta, columns=4)
    indices = np.flatnonzero(np.isfinite(z.ravel()))
    sample = indices[np.linspace(0, len(indices)-1, 7, dtype=int)]
    provenance = {"protocol":report["protocol"], "input_sha256":meta["input_sha256"],
                  "posterior_sha256":report["full_packet_sha256"], "volume_km3":report["volume_km3"],
                  "median_map_volume_km3":report["median_map_volume_km3"], "map_draws":len(shown.z),
                  "x_km_y_km_log10_median_cm":np.column_stack((xy[sample], z.ravel()[sample])).tolist()}
    fig.savefig(out/"isopachs.png", dpi=150, bbox_inches="tight",
                metadata={"Description":json.dumps(provenance, sort_keys=True)})
    plt.close(fig)


def both_maps(data, meta, name, out, model, previous):
    x, y = grid_axes(frame_bounds(data.xy), longest=181)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    views = [MapView(m.reduced(512)) for m in (previous, model)]
    fields = [v.central(points).reshape(xx.shape) for v in views]
    norm = Normalize(-2., max(2., np.ceil(np.nanmax(fields))))
    mesh = SupportMesh(data)
    both_records = []
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9), layout="constrained")
    for ax, z, deposit, title in zip(axes, fields, (previous, model), ("Whole-area rank", "Local dispersal corridors"), strict=True):
        image = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis", norm=norm, rasterized=True)
        rows = contour_segments(ax, x, y, z, data, mesh, deposit)
        both_records.extend((title, *row) for row in rows)
        setup_map(ax, frame_bounds(data.xy), meta["vent_lonlat"])
        observations(ax, data, size=16, meta=meta)
        ax.set_title(title)
    thickness_bar(fig, image, axes, norm)
    fig.suptitle(f"{name} · Median isopachs (cm)\nSolid where measurements bracket the contour", fontsize=14)
    observation_legend(fig, data, meta=meta)
    fig.savefig(out / "paired-isopachs.png", dpi=145)
    plt.close(fig)
    with io.StringIO(newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model","level_cm","triangle","vertex_1","vertex_2","vertex_3","spacing_accepted",
                         "measured_bracket","posterior_crossing_probability","display_solid"))
        writer.writerows(both_records)
        (out / "local-brackets.csv.gz").write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
    world_map(out, name + " · Local dispersal corridors", data, meta, views[1], norm)
    compass_map(out, name, data, model, previous)


def paired_areas(data, model, previous):
    """Same median-field statistic and identical integration sites before and after."""
    mesh = SupportMesh(data)
    parts = [previous.reduced(768), model.reduced(768)]
    runs = []
    for seed in (28439, 95717):
        xy, w, r, theta = nodes(15, seed, model.latitude)
        w /= len(r)
        sector = np.floor((theta + np.pi / 8) / (np.pi / 4)).astype(int) % 8
        tri = mesh.triangulation.find_simplex(xy)
        outside = ~((tri >= 0) & mesh.accepted[np.maximum(tri, 0)])
        result = np.zeros((2, len(LEVELS), 9))
        for ids in np.array_split(np.arange(len(r)), 144):
            for i, m in enumerate(parts):
                mu = np.median(m.field(xy[ids]), axis=0)
                areas = (mu >= np.log(LEVELS[:, None])) * w[ids]
                for k in range(8):
                    result[i, :, k] += areas[:, sector[ids] == k].sum(axis=1)
                result[i, :, 8] += areas[:, outside[ids]].sum(axis=1)
        runs.append(result)
    return {"level_cm":LEVELS.tolist(), "columns":[*SECTORS, "outside_retained_exact_triangles"],
            "method_order":["whole_area_rank", "corridor"], "nodes_per_scramble":2**15,
            "draws_per_map":[len(m.z) for m in parts], "scrambles":[28439,95717],
            "area_km2":np.mean(runs, axis=0).tolist(), "by_scramble_km2":np.asarray(runs).tolist()}


def prior_sensitivity(budgets):
    from ..directional.ensemble.fields import inverse_cdf
    volume, law = budgets["volume_km3"], budgets["law"]
    scenarios = {}
    for broad in (.10, .25, .50, .75):
        weights = np.where(law == 0, broad / PRIOR[0], (1.-broad) / (1.-PRIOR[0]))
        weights /= weights.sum()
        quantiles = np.quantile(volume, PROBS) if broad == .25 else inverse_cdf(volume, weights, PROBS)
        scenarios[str(broad)] = {"volume_km3":quantiles.tolist(),
                                 "effective_draws":float(1. / (weights @ weights))}
    conditional = []
    for i,(width,extension) in enumerate(LAWS):
        choose = law == i
        n = int(choose.sum())
        conditional.append({"width_deg":width,"extension":extension,"joint_draws":n,
                            "volume_km3":np.quantile(volume[choose], PROBS).tolist() if n >= 40 else None})
    return {"broad_prior_scenarios":scenarios, "conditional_by_law":conditional}


def heldout_coverage(model, test):
    """Central equal-tail predictive coverage of exact readings under the fitted observation law."""
    exact = test.subset(~test.censored) if (~test.censored).any() else None
    if exact is None:
        return {"exact_rows":[], "predictive_cdf":[], "covered":0}
    residual = exact.log_value - model.field(exact.xy)
    scatter = parameters(model.z)[-1][:, None]
    ranks = (.95*ndtr(residual/np.hypot(scatter, exact.log_sd))
             + .05*ndtr(residual/np.hypot(3.*scatter, exact.log_sd))).mean(axis=0)
    return {"exact_rows":exact.row_ids.tolist(), "predictive_cdf":ranks.tolist(),
            "covered":int(((ranks >= .025) & (ranks <= .975)).sum())}


def set_principal_summary(report):
    groups = [f["exact_site_coverage95"] for f in report["folds"]]
    count = sum(len(g["exact_rows"]) for g in groups)
    covered = sum(g["covered"] for g in groups)
    report.update(model_role="principal", conditional_volume_km3=report["volume_km3"],
                  volume_statistic="median of complete Earth-integrated posterior draws",
                  map_statistic="pointwise posterior median thickness",
                  exact_site_coverage95={"covered":covered, "exact_sites":count,
                                         "fraction":covered/count if count else None})


def case_text(report, *, whole_area_link=None):
    lo, med, hi = report["volume_km3"]
    plo, pmed, phi = report["reference_volume_km3"]
    nums = report["numerics"]
    score, prior = report["geographic_score"], report["reference_geographic_score"]
    lines = [f"# {report['name']} · Bayesian nested isopachs", "",
             f"The principal corridor estimate of bulk fallout is **{med:,.1f} km³** (conditional 95% interval **{lo:,.1f}–{hi:,.1f}**). "
             f"The median thickness map integrates to {report['median_map_volume_km3']:,.1f} km³. "
             f"The whole-area sensitivity gives {pmed:,.1f} km³ ({plo:,.1f}–{phi:,.1f}).", "",
             "![Principal corridor median thickness, measured sites and local contour support](isopachs.png)", "",
             "Solid portions cross a retained triangle with thicknesses on either side at distinct measured sites and at least 90% posterior crossing probability. Dashed portions depend on the continuation. "
             "The triangles diagnose spatial coverage. The shared contour screen also checks thickness bracketing and deletion stability. "
             "[Whole-Earth field](world.png) · [Every local bracket](local-brackets.csv.gz) · [Complete contour checks](support.json)", "",
             f"The buffered geographic score is {score:.3f} for the corridor and {prior:.3f} for the whole-area sensitivity. Lower scores indicate better predictions at withheld sites. "
             f"The corridor’s 95% predictive intervals cover {report['exact_site_coverage95']['covered']} of {report['exact_site_coverage95']['exact_sites']} exact withheld readings. "
             f"{sum(f['passed'] for f in report['folds'])} of {len(report['folds'])} geographic fits passed their sampling checks. "
             f"The full-data proposal and importance check {'passed' if nums['fit'] else 'retained a warning'}, the final volume populations "
             f"{'passed' if nums['populations'] else 'retained a warning'}, and the integration check {'passed' if nums['quadrature'] else 'retained a warning'}. "
             f"The shared screen accepted a complete contour band under {len(report['supported_band_policies'])} policies. "
             "[Prediction scores and row identities](report.json) · [Full fit diagnostics](full.json)", "",
             "## Where the volume lies", "",
             "![Full compass, measured values, median contours and conditional uncertainty](compass.png)", "",
             "The full-compass view includes all bearings through the antipode. Its logarithmic distance scale makes the proximal flanks visible. Areas in the table use the physical equal-area integral.", "",
             "| Compass sector | Exact sites | Bounded sites | Corridor 1 cm area (km²) | Whole-area 1 cm area (km²) | Corridor volume (km³), median [95%] |",
             "|---|---:|---:|---:|---:|---:|"]
    areas = np.asarray(report["paired_areas"]["area_km2"])[:, 2]
    def volume_label(value):
        return f"{value:.2g}" if 0 < value < .1 else f"{value:,.1f}"

    for j, (name, (vlow, centre, vhigh)) in enumerate(zip(SECTORS, report["sector_volume_km3"], strict=True)):
        exact, bounded = report["site_counts_by_sector"][j]
        lines.append(f"| {name} | {exact} | {bounded} | {areas[1,j]:,.0f} | {areas[0,j]:,.0f} | {volume_label(centre)} [{volume_label(vlow)}–{volume_label(vhigh)}] |")
    lines.extend(["", f"At 1 cm, the corridor median map encloses {areas[1,:8].sum():,.0f} km² and the whole-area sensitivity encloses {areas[0,:8].sum():,.0f} km². "
                  f"Of these, {areas[1,8]:,.0f} and {areas[0,8]:,.0f} km² fall outside the retained exact-site triangles. "
                  "An unsampled sector can contain ash. Its allocation depends on the assumed continuation.", "",
                  "The prior assigns 25% to continuation across all bearings. The remaining probability spans three fan widths and three down-range extensions. "
                  f"The fitted probability of whole-bearing continuation is {report['law_posterior'][0]:.1%}. "
                  "The widths and extensions are shared structural assumptions. A bound supplies a likelihood event. "
                  "Only positive evidence supplies a potential dispersal corridor.", "",
                  "[Volume and sector draws](volume-budget.npz) · [Complete field draws](full.npz) · [Areas, assumptions, bounds and diagnostics](report.json)"])
    lines.extend(["", "## Structural uncertainty", "",
                  "[Corridor and whole-area maps on a common scale](paired-isopachs.png)", "",
                  "| Prior probability of broad continuation | Median bulk volume, km³ [conditional 95%] |",
                  "|---:|---:|"])
    for p, result in report["prior_sensitivity"]["broad_prior_scenarios"].items():
        left, centre, right = result["volume_km3"]
        lines.append(f"| {float(p):.0%} | {centre:,.1f} [{left:,.1f}–{right:,.1f}] |")
    lines.extend(["", "The prior odds vary under the same observed likelihood. The numerical report also gives the volume by individual corridor law where the saved draws resolve it."])
    for b in report["bounds"]:
        if b["name"] == "Lake Sihailongwan":
            lines.extend(["", f"At Lake Sihailongwan the fitted median is {b['latent_cm'][1]:.3g} cm and the predictive probability of the interpreted ≤{b['upper_cm']:.2g} cm event is {b['predictive_probability_of_event']:.2%}. "
                          "The archive constrains an upper limit. The actual bed thickness remains unknown."])
    if whole_area_link:
        lines.extend(["", f"[Whole-area sensitivity report, posterior and diagnostics]({whole_area_link})"])
    return "\n".join(lines) + "\n"


def reach_dataset(path, output, name=None, *, parent=None, resume=False, archives=None):
    out = Path(output)
    if out.is_dir() and any(out.iterdir()) and not resume:
        raise ValueError("Use a fresh corridor result directory or verified resume")
    out.mkdir(parents=True, exist_ok=True)
    data, meta = audited_data(path, allow_intervals=True)
    if resume:
        for key in ("full", "fold-0", "fold-1", "fold-2", "fold-3"):
            info, packet = out/(key+".json"), out/(key+".npz")
            if info.exists() or packet.exists():
                row = json.loads(info.read_text()) if info.exists() else {}
                ident = row.get("identity", {})
                if (ident.get("protocol") != PROTOCOL or ident.get("input_sha256") != meta["input_sha256"]
                        or ident.get("engine_sha256") != signature() or not packet.is_file()
                        or row.get("posterior_sha256") != digest(packet)):
                    raise ValueError(f"Resume inputs or model differ from {info}. Choose a fresh output directory.")
    name = name or (dataset_record(path) or {}).get("name", Path(path).stem)
    if parent is None:
        parent = out / "whole-area-parent"
        base = out / "intrinsic"
        infer_dataset(path, base)
        from_parent(data, meta, base, parent)
    parent = Path(parent)
    lat = meta["vent_lonlat"][1]
    full, receipt = conditioned_fit(data, None, meta, parent, out, "full", 71183)
    previous = load_packet(parent / "full.npz")
    report, budgets = checked(full, data)
    np.savez_compressed(out / "volume-budget.npz", **budgets)
    folds, deletions = [], []
    for i, train, test in splits(data, lat, "geographic"):
        model, row = conditioned_fit(train, test, meta, parent, out, f"fold-{i}", 71184+i)
        folds.append({"fold":i, "passed":row["passed"], "training_rows":train.row_ids.tolist(),
                      "test_rows":test.row_ids.tolist(), "row_log_scores":row["row_log_scores"],
                      "exact_site_coverage95":heldout_coverage(model, test),
                      "reference_row_log_scores":row["reference_row_log_scores"],
                      "negative_log_score":float(-np.mean(row["row_log_scores"])),
                      "reference_negative_log_score":float(-np.mean(row["reference_row_log_scores"]))})
        if row["passed"]:
            deletions.append(model.reduced(256))
    bounds = survey_bounds(data.xy)
    corners = np.array([[a,b] for a in bounds[[0,2]] for b in bounds[[1,3]]])
    if EarthDomain.wgs84(lat).rho(corners).max() < DIAMETER:
        screen, *_ = audit_contours(full.reduced(256), data, bounds, deletions)
        screen["status"] = "completed"
    else:
        screen = {"status":"audit_window_leaves_physical_chart", "policies":{}}
    screen["numerically_usable_geographic_folds"] = [f["fold"] for f in folds if f["passed"]]
    write_json(out / "support.json", screen)
    from .geometry import positions
    theta = positions(data.xy, lat)[1]
    sector = np.floor((theta+np.pi/8)/(np.pi/4)).astype(int) % 8
    last = receipt["importance_attempts"][-1]
    report.update(protocol=PROTOCOL, name=name, input_sha256=meta["input_sha256"], engine_sha256=signature(),
                  full_packet_sha256=digest(out / "full.npz"), reference_packet_sha256=digest(parent / "full.npz"),
                  reference_volume_km3=np.quantile(previous.volume(), PROBS).tolist(),
                  law_order=LAWS, law_prior=PRIOR.tolist(),
                  law_posterior=np.mean([p["law_probabilities"] for p in last["populations"]],axis=0).tolist(),
                  folds=folds, geographic_score=float(np.mean([f["negative_log_score"] for f in folds])),
                  reference_geographic_score=float(np.mean([f["reference_negative_log_score"] for f in folds])),
                  supported_band_policies=[k for k,p in screen["policies"].items() if p["selected_band"]],
                  support_status=screen["status"], bounds=check_bounds(data, meta, full),
                  paired_areas=paired_areas(data, full, previous),
                  site_counts_by_sector=[[int(sum((sector==j) & ~data.censored)),int(sum((sector==j) & data.censored))] for j in range(8)],
                  numerics={"fit":receipt["passed"],"populations":report["volume_populations_passed"],
                            "quadrature":report["quadrature"][-1]["passed"]})
    report["archive_predictions"] = archive_predictions(path, archives, data, meta, full)
    report["prior_sensitivity"] = prior_sensitivity(budgets)
    set_principal_summary(report)
    write_json(out / "report.json", report)
    principal_map(data, meta, name, out, full, report)
    both_maps(data, meta, name, out, full, previous)
    sibling = None
    if parent.resolve() == (out.parent/"nested-whole-area").resolve() and (parent/"README.md").is_file():
        sibling = "../nested-whole-area/README.md"
    (out / "README.md").write_text(case_text(report, whole_area_link=sibling))
    return report
