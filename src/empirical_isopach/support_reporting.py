"""Auditable post-fit support diagnostics, separate from family tuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import covers, points, get_parts
from shapely.geometry import mapping

from .contour_support import audit_contours
from .support import POLICIES, SupportMesh
from .region import integrate_region
from .core.validation import interval_metrics
from .models.registry import method_label


def write_support_analysis(output, name, data, bounds, models, deletion_models, validation,
                           base_provenance, progress=print, resolution=257):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meshes = {p.name: SupportMesh(data, p) for p in POLICIES}
    report = {"dataset": name, "protocol": "supported-isopachs-v1",
              "provenance": base_provenance, "fit_bounds_km": list(map(float, bounds)),
              "support": {k: v.describe() for k, v in meshes.items()}, "methods": {},
              "preferred_global_km3": None,
              "interpretation": "Local footprints and contour eligibility are prespecified development screens, not calibrated confidence sets. All policies and failed contours are retained. No observations or fit parameters are changed."}
    maps, rows = {}, []
    for method, model in models.items():
        deletion = deletion_models[method]
        volumes = {}
        for key, mesh in meshes.items():
            if mesh.area <= 0:
                volumes[key] = {"status": "empty_support", "expected_km3": None}
                continue
            volume = integrate_region(model, mesh)
            removed = [integrate_region(m, mesh) for m in deletion]
            volume["deletion_expected_km3"] = [v["expected_km3"] for v in removed]
            volume["deletion_integrals_converged"] = all(v["converged"] for v in removed)
            volume["uncertainty"] = "Geographic-deletion sensitivity only; not a confidence interval. Common footprint held fixed. Selection-adjusted volume intervals are not supplied."
            volumes[key] = volume
        contour, grid, polygons, deletion_grids = audit_contours(model, data, bounds, deletion, meshes, resolution)
        # Use training geometry only for out-of-fold support labels. Never use
        # the withheld site's response or add its coordinate to the support mesh.
        stratum_masks = {k: np.zeros(len(data.xy), bool) for k in meshes}
        splits = validation["methods"][method]["selections"]
        for split in splits:
            train = np.isin(data.row_ids, split["training_row_ids"])
            test = np.isin(data.row_ids, split["withheld_row_ids"])
            for key, mesh in meshes.items():
                training_mesh = SupportMesh(data.subset(train), mesh.policy)
                stratum_masks[key][test] = covers(training_mesh.geometry, points(data.xy[test]))
        strata = {}
        predicted = validation["methods"][method]["predictions"]
        arrays = {k: np.asarray(v) for k, v in predicted.items()}
        for key, mask in stratum_masks.items():
            strata[key] = {}
            for label, chosen in (("inside_training_support", mask), ("outside_training_support", ~mask)):
                sub = data.subset(chosen)
                metrics = None
                if (~sub.censored).any():
                    metrics = interval_metrics(sub, arrays["median"][chosen], arrays["lower"][chosen],
                                               arrays["upper"][chosen], arrays["log_score"][chosen])
                strata[key][label] = {"n": int(chosen.sum()), "n_exact": int((~sub.censored).sum()),
                    "row_ids": sub.row_ids.tolist(), "metrics": metrics}
        report["methods"][method] = {"label": method_label(method), "footprint_volumes": volumes,
            "contours": contour, "level_density_sensitivity": contour["level_density_sensitivity"],
            "withheld_support_strata": strata}
        maps[method] = (grid, polygons)
        for row in contour["levels"]:
            for key, details in row["policies"].items():
                rows.append({"method": method, "policy": key, **{k: v for k, v in row.items()
                    if k not in ("policies", "bracketing", "deletion_shape_changes")},
                    **details, "reasons": ";".join(details["reasons"]),
                    "deletion_shape_changes": json.dumps(row.get("deletion_shape_changes", []))})
        progress(f"{name}: {method_label(method)} support, contour and extrapolation audit complete")
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    pd.DataFrame(rows).to_csv(output/"contour_audit.csv", index=False)
    (output/"support_geometry.json").write_text(json.dumps({"type": "FeatureCollection",
        "coordinate_note": "Source-centred LAEA kilometres, NOT longitude/latitude; see report fit/input provenance.",
        "features": [{"type": "Feature", "properties": {"policy": key, "area_km2": mesh.area},
                      "geometry": mapping(mesh.geometry)} for key, mesh in meshes.items()]}, allow_nan=False)+"\n")
    support_figures(output, name, data, meshes, maps, report)
    support_readme(output, name, report)
    return report


def support_figures(output, name, data, meshes, maps, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src.probabilistic_isopach.figures import STYLE, TEAL, AMBER, INK
    mesh = meshes["standard"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(maps), figsize=(6.7*len(maps), 6), layout="constrained", squeeze=False)
        for ax, (method, (grid, polygons)) in zip(axes[0], maps.items()):
            for triangle in mesh.triangles:
                ax.fill(*triangle.T, color="#D9EBE7", lw=0, zorder=0)
            for part in get_parts(mesh.geometry):
                ax.plot(*part.exterior.xy, color=TEAL, lw=.8)
                for hole in part.interiors:
                    ax.plot(*hole.xy, color=TEAL, lw=.8)
            records = report["methods"][method]["contours"]["levels"]
            by_level = {r["level_cm"]: r["policies"]["standard"]["eligible"] for r in records}
            x, y, z = grid
            levels = [np.log(r["level_cm"]) for r in records]
            if levels:
                ax.contour(x, y, z, levels=levels, colors="#ADAFB2", linewidths=.55, linestyles="dashed")
            for level, polygon in polygons.items():
                if not by_level.get(level, False):
                    continue
                for part in get_parts(polygon):
                    ax.plot(*part.exterior.xy, color=TEAL, lw=1.6)
                    for hole in part.interiors:
                        ax.plot(*hole.xy, color=TEAL, lw=1.6)
            for mask, marker, color in ((~data.censored, "o", INK), (data.censored, "^", AMBER)):
                ax.scatter(*data.xy[mask].T, marker=marker, s=12, facecolors="white", edgecolors=color, lw=.6, zorder=5)
            n = sum(by_level.values())
            ax.set(aspect="equal", xlabel="Easting (km)", ylabel="Northing (km)",
                title=f"{method_label(method)}\n{n} / {len(records)} contours pass every standard check")
            ax.grid(alpha=.1)
        axes[0, 0].legend(handles=[Line2D([], [], color=TEAL, lw=6, alpha=.25, label="Local interpolation footprint"),
            Line2D([], [], color="#ADAFB2", ls="--", label="Unqualified model contour"),
            Line2D([], [], color=TEAL, lw=1.6, label="Qualified complete contour"),
            Line2D([], [], color=INK, marker="o", markerfacecolor="white", ls="", label="Exact measurement"),
            Line2D([], [], color=AMBER, marker="^", markerfacecolor="white", ls="", label="Documented bound")], fontsize=8, loc="lower left")
        fig.suptitle(f"{name} · closure is not evidence\nGaps stay open; rejected curves do not define global volume.", fontsize=15)
        fig.savefig(output/"support.png", dpi=170)
        plt.close(fig)


def support_readme(output, name, report):
    methods = report["methods"]
    lines = [f"# {name}: which parts of the deposit are supported?", "",
        "**A closed or smooth isopach is not sufficient evidence for its enclosed area or a global volume.** This audit applies the same geometric screen to both model families, then checks their contours separately. It does not change the observations or refit to pseudo-data.", "",
        "![Sampling support and contour eligibility](support.png)", "",
        "## Local interpolation volumes", "",
        "The common footprint keeps well-connected triangles between exact measurement sites and removes long bridges and thin triangles. Its holes remain holes. Bounds still enter both model likelihoods, but do not count as exact thickness sites for this geometric screen.", "",
        "| Policy | Area (km²) | Method | Expected bulk volume (km³) | Median-surface integral (km³) | Deletion range (km³) |",
        "|---|---:|---|---:|---:|---:|"]
    for key, mesh in report["support"].items():
        for method, result in methods.items():
            v = result["footprint_volumes"][key]
            if v.get("expected_km3") is None:
                values = "— | — | empty footprint"
            else:
                deleted = v["deletion_expected_km3"]
                values = f"{v['expected_km3']:.2f} | {v['median_surface_km3']:.2f} | {min(deleted):.2f}–{max(deleted):.2f}"
            lines.append(f"| {key} | {mesh['area_km2']:,.0f} | {result['label']} | {values} |")
    lines += ["", "These are **partial bulk fallout volumes**, not complete eruption volumes, dense-rock equivalents or certified lower bounds. The footprint depends on the sampling design and an explicit spacing rule. The three policies are sensitivity analyses; their range and the four geographic-deletion values are **not confidence intervals**. The median-surface integral is not the median of total volume.", "",
        "## Complete-contour gate", "",
        "Every candidate must close without touching the calculation boundary, have its entire perimeter inside the local footprint, be bracketed by at least three distinct exact sites on each required side, converge under map refinement, and remain closed and shape-stable after each of four fully retuned geographic deletions. All islands and holes count. A usable band also needs three consecutive qualifying contours spanning at least a factor of two in thickness, with no unsupported area between them.", "",
        "| Method | Policy | Passing contours / 25 | Usable bands | Extension status |",
        "|---|---|---:|---:|---|"]
    for method, result in methods.items():
        contour = result["contours"]
        for key, policy in contour["policies"].items():
            passing = sum(r["policies"][key]["eligible"] for r in contour["levels"])
            bands = sum(r["eligible_for_extension"] for r in policy["contiguous_runs"])
            lines.append(f"| {result['label']} | {key} | {passing} / {len(contour['levels'])} | {bands} | {policy['extension']['status']} |")
    lines += ["", "Eligibility is a conservative development rule, not a 95% confidence statement. Spacing factors 1.5, 2 and 3 and deletion-shape tolerances 10%, 25% and 50% are fixed for every eruption. A 49-level sensitivity is recorded in the JSON. No failing threshold is relaxed for a particular eruption.", "",
        "## Extrapolation and pseudo-data", "",
        "No preferred global volume is reported by this audit. Where a band qualifies, its geographic volume is integrated directly, with a separate layer-cake check; endpoint-matched proximal and distal continuations are conditional scenarios only. Nonintegrable power tails are refused rather than cut off at an arbitrary distance. Where no band qualifies, no continuation is fitted.", "",
        "The existing spline trend/derivative penalties regularize unsampled space; their numerical quadrature is not extra evidence. Neither those penalties nor GP mean reversion validate closure. Missing sites are not zeros, predictions are not observations, and multiple curves derived from one surface are not independent data.", "",
        "## Withheld checks and numerical limitations", ""]
    for method, result in methods.items():
        standard = result["withheld_support_strata"]["standard"]
        counts = [standard[k]["n_exact"] for k in ("inside_training_support", "outside_training_support")]
        lines.append(f"- {result['label']}: {counts[0]} exact outer-test sites inside training-only support; {counts[1]} outside. See the JSON for separate errors/coverage and row identities; sparse strata do not establish calibration.")
        for key, volume in result["footprint_volumes"].items():
            if volume.get("expected_km3") is not None and (not volume["converged"] or not volume["deletion_integrals_converged"]):
                lines.append(f"- {result['label']} / {key}: integral refinement did not converge; this result needs numerical refinement before quantitative use.")
    lines += ["", "The contour gate uses fitted responses and model deletion stability. Its selected band has no selection-adjusted confidence interval. Geographic holdouts test prediction at observed sites, not an unobserved distal tail. Even smooth positive finite-mass fields can agree at every sampled site yet have arbitrarily different missing volume.", "",
        "[Every contour and rejection reason](contour_audit.csv) · [Full numerical and provenance record](report.json) · [Original matched-hull comparison](../README.md)"]
    (output/"README.md").write_text("\n".join(lines)+"\n")
