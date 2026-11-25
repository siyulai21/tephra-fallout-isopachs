"""Publish one global median deposit, its physical integral and its thickness–area curve."""
import csv
import json
from pathlib import Path

import numpy as np

from .deposit_field import FAMILIES, PolarField, digest, display_name, doubling_levels, open_field, review_path, sample_field
from .deposit_integrals import independent_integral, within_cells
from .deposit_field import write_map_grid
from .deposit_maps import LAND, frames, outer_contour, plate
from .maps import boundary_points
from .contour_limits import inner_contour, near_frame
from .thickness_band import checked_band, median_trials


def volume_interval(family, report, extent=None):
    """Recover whole-draw endpoints for historical comparisons and fit diagnostics."""
    if "independent_draw_integration" in report:
        return tuple(report["conditional_interval95_km3"]), report["draw_checks_passed"]
    if extent is not None:
        return tuple(extent["conditional_interval95_km3"]), extent["draw_checks_passed"]
    if family=="nested-isopachs":
        q = report.get("conditional_volume_interval95_km3") or report["volume_km3"]
        return (q[0], q[-1]), all(report["numerics"].values())
    if family=="directional":
        vol = report["volume"]
        q = vol["coherent_volume_km3"]
        ok = (all(m["diagnostics"]["passed"] for m in report["members"].values())
              and vol["numerical"]["joint_quantile_split_passed"])
    else:
        fit = report["fits"]["baseline"]
        q, ok = fit["volume"],fit["diagnostics"]["passed"]
    return (q["lower95"], q["upper95"]), ok


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def deposit_page(old, family, name, receipt, report):
    lo,_,hi = receipt["thickness_band"]["volumes_km3"]
    border = receipt["outer_contour"]
    warning = "" if receipt["posterior_checks_passed"] else " **Numerical warning on the fitted deposit distribution.**"
    volume=receipt["median_map_volume_km3"]
    volume_label=f"{volume:,.0f}" if volume>=1000 else f"{volume:,.3g}"
    label = display_name(family, receipt["sources_sha256"])
    first = [f"# {name} · {label}", "",
             f"The principal bulk volume is **{volume_label} km³**, the whole-Earth integral of the median-thickness map. "
             f"The conditional thickness-band range is **{lo:,.3g}–{hi:,.3g} km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth.{warning}", "",
             f"![{name}, {label}. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)", "",
             "Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. "
             f"The {border['level_cm']:g} cm footprint includes {border['inside']} of {border['accepted_rows']} modeled locations. "
             f"The thickest displayed line is {receipt['inner_contour']['level_cm']:g} cm and encloses {receipt['inner_contour']['inside']} fitted locations. "
             "The volume includes fallout beyond the displayed contours. "
             "[Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)", ""]
    if receipt.get("historical_input"):
        first[2:2] = [f"**Historical comparison on {border['accepted_rows']} localities.** The eruption catalogue uses the latest audited input.", ""]
    if not receipt["integration_passed"] or not receipt["thickness_band"]["passed"]:
        first.extend(["**Map integration warning.** The independent or refinement check exceeded its tolerance. See the numerical record.", ""])
    if "multiscale/report.json" in receipt["sources_sha256"]:
        v=report["validation"]["geographic"]
        first.extend([f"The equal-region negative log score was {v['negative_log_score']:.3f}. "
                      f"The 95% predictive intervals covered {v['coverage95_exact']:.1%} of the withheld measurements. "
                      f"{v['passed_folds']} of four buffered fits passed the numerical checks.", "",
                      "The directional field allows fan headings and contrast to change with distance. Nearby stations can support local departures. "
                      "Solid line fragments require exact measurements on both sides and at least 90% posterior probability of a crossing.", "",
                      f"The conditional 95% interval for integrated whole-deposit draws is {report['conditional_interval95_km3'][0]:,.3g}–{report['conditional_interval95_km3'][1]:,.3g} km³. "
                      f"{len(report['support_policies_with_band'])} support policies found a qualifying complete-contour band.", "",
                      "[Fit, structural sensitivities and convergence](multiscale/report.json) · [All contour decisions](multiscale/support.json.gz) · "
                      "[Measured sites](multiscale/measured-sites.csv.gz) · [Held-out predictions](multiscale/heldout.csv.gz) · [Archive locations](multiscale/archive-probes.json) · [Thickness exceedance at three scales](exceedance.png)"])
        return "\n".join(first)+"\n"
    if "footprints/report.json" in receipt["sources_sha256"]:
        v = report["validation"]
        first.extend([f"The fitted model predicts the four buffered geographic groups with an equal-group negative log score of {v['negative_log_score']:.3f}. "
                      f"{v['passed_folds']} of four independent deletion fits passed their numerical checks.", "",
                      "Each deposit draw expands one or more dispersal footprints. Stations enter through their thickness evidence. "
                      "The direction and extent of each phase are inferred jointly with the area–thickness curve.", "",
                      "[Joint fit, structural sensitivities and numerical checks](footprints/report.json) · "
                      "[Held-out predictions](footprints/heldout.csv.gz) · [Contour support and all rejection reasons](footprints/support.json.gz) · "
                      "[Archive-site predictions](footprints/archive-probes.json) · [Volume and observations by compass sector](footprints/sector-volumes.csv)"])
        return "\n".join(first)+"\n"
    if "extent/report.json" in receipt["sources_sha256"]:
        markers = {"directional":"## Reach from the volcano","spherical-mixture":"## What the unobserved directions contribute",
                   "nested-isopachs":"The buffered geographic score is"}
        pivot = markers[family]
        _,found,tail = old.partition(pivot)
        if not found:
            raise ValueError("Cannot locate the parent diagnostics")
        first.extend(["The principal map uses the geographic continuation evaluated on all registered eruptions. "
                     "[Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · "
                     "[Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)", "",
                     "## Earlier continuation and its diagnostics", "",
                     "The following figures and statistics describe the fitted parent and its broader continuation. "
                     "The principal estimate and its interval appear above.", ""])
        if family=="spherical-mixture":
            index=first.index("## Earlier continuation and its diagnostics")
            first[index:index]=["[Conditional intervals under all eight fixed prior fits](extent/prior-sensitivity.json)", ""]
        return "\n".join(first)+"\n"+pivot+tail
    if family=="directional":
        pivot = "## Reach from the volcano"
        vol = report["volume"]
        first += [f"Fixing the fan directions gives a median-map volume of {vol['matched_control_map_km3']:,.3g} km³. "
                  "Solid fragments require thicknesses on opposite sides of the contour and at least 90% posterior crossing probability. "
                  "[Whole-Earth context](azimuth/world.png) · [Earlier 0.1 cm extent and its edge checks](azimuth/map-display.json)", ""]
    elif family=="spherical-mixture":
        pivot = "## What the unobserved directions contribute"
        first += ["Every contour on these maps is conditional. [Spatial uncertainty and measurement footprint](field.png)", ""]
    else:
        pivot = "The buffered geographic score is"
        first += ["Solid segments cross locally bracketed triangles with at least 90% posterior crossing probability. "
                  "[Whole-Earth context](world.png) · [Local brackets](local-brackets.csv.gz) · [Complete contour checks](support.json)", ""]
    _, found, tail = old.partition(pivot)
    if not found:
        raise ValueError(f"Cannot locate the diagnostic section of the {family} report")
    return "\n".join(first) + "\n" + pivot + tail


def describe_surface(field, levels, checks, bounds, outer_cm=.5):
    volume,area=within_cells(field,levels)
    lower_order,_=within_cells(field,order=2,angles=2)
    coarse=PolarField(field.radii[::2],field.log10_cm[::2,::2],field.axis)
    coarse_volume,coarse_area=within_cells(coarse,levels)
    edge=boundary_points(bounds)
    edge_cm=10.**field.at_xy(*edge.T)
    edge_max=float(np.nanmax(edge_cm)) if np.isfinite(edge_cm).any() else None
    if edge_max is not None and edge_max>=outer_cm:
        raise ArithmeticError(f"The full-deposit map clips the {outer_cm:g} cm contour")
    probe=zip(np.linspace(0,len(field.radii)-2,13).astype(int),np.linspace(0,len(field.angles)-1,13).astype(int),strict=True)
    points=[[field.radii[i]*np.sin(field.angles[j])*field.axis,field.radii[i]*np.cos(field.angles[j])/field.axis,
             float(field.log10_cm[i,j])] for i,j in probe]
    reference=float(np.mean(checks[-1]["km3"]))
    record=dict(grid_shape=list(field.log10_cm.shape),independent_integrals=checks,coarse_grid_volume_km3=coarse_volume,
                lower_quadrature_order_volume_km3=lower_order,field_replay_relative_difference=abs(volume/reference-1),
                integration_passed=bool(abs(volume/reference-1)<.015 and abs(coarse_volume/volume-1)<.02
                    and abs(lower_order/volume-1)<.001 and checks[-1]["relative_change"]<.01),
                maximum_display_edge_cm=edge_max,antipodal_ring_maximum_cm=float(10.**np.max(field.log10_cm[-1])),
                check_points_x_y_km_log10_cm=points)
    return volume,area,coarse_area,record


def checked_field(view,data,latitude):
    field=sample_field(view,data,latitude)
    checks=independent_integral(view,view.domain)
    levels=doubling_levels(10.**field.log10_cm.max())
    guess,area=within_cells(field,levels)
    coarse=PolarField(field.radii[::2],field.log10_cm[::2,::2],field.axis)
    _,coarse_area=within_cells(coarse,levels)
    sensitive=np.any(abs(area-coarse_area)>np.maximum(1.,.05*area))
    if sensitive or abs(guess/np.mean(checks[-1]["km3"])-1)>.012:
        field=sample_field(view,data,latitude,385,480)
    return field,checks


def render_deposit(folder, family, data, meta, name, *, output=None, sampled=None, near_half=None):
    """Integrate and draw the median thickness of the saved posterior."""
    folder=Path(folder)
    dest=Path(output) if output else folder
    dest.mkdir(parents=True,exist_ok=True)
    review = review_path(dest, family)
    if not review.is_file():
        review = review_path(folder, family)
    view,model,sources=open_field(folder,family,data,meta,review=review)
    latitude=meta["vent_lonlat"][1]
    if sampled is None:
        field,band=checked_band(view,data,latitude)
    else:
        field,band,identity=sampled
        if identity!=sources:
            raise ValueError("Scratch field belongs to different saved fits")
    report=read_json(folder/next(iter(sources)))
    _,fit_ok=volume_interval(family,report,read_json(review) if review.exists() else None)
    maximum=max(field.log10_cm.max(),data.log_value.max()/np.log(10.))
    levels=doubling_levels(10.**maximum)
    border=outer_contour(field,data)
    bounds,_,_=frames(field,data,border["level_cm"])
    volume,area,coarse_area,checks_record=describe_surface(field,levels,median_trials(band),bounds,border["level_cm"])
    np.testing.assert_allclose(volume,band["volumes_km3"][1],rtol=1e-12)
    receipt=dict(protocol="median-thickness-deposit-v2",family=family,eruption=name,domain="WGS84 whole Earth",
                 input_sha256=meta["input_sha256"],sources_sha256=sources,land_sha256=digest(LAND),
                 median_map_volume_km3=volume,thickness_band=band,outer_contour=border,inner_contour=inner_contour(border),
                 posterior_checks_passed=bool(fit_ok),**checks_record)
    from ..core.catalog import datasets
    if any(meta.get("source")=="data/"+old["file"] for rec in datasets() for old in rec.get("previous_inputs",[])):
        receipt["historical_input"] = True
    rows=[dict(thickness_cm=cm,area_km2=a,sqrt_area_km=np.sqrt(a),coarse_area_km2=b,
               resolution_status=("no sampled exceedance" if a==0 else "grid sensitive" if abs(a-b)>max(1.,.05*a) else "resolved"))
          for cm,a,b in zip(levels,area,coarse_area,strict=True)]
    with (dest/"thickness-area.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_map_grid(dest/"median-grid.npz",field,model,family,meta,sources)
    receipt["median_grid_sha256"]=digest(dest/"median-grid.npz")
    receipt["thickness_area_sha256"]=digest(dest/"thickness-area.csv")
    plate(dest/"deposit.png",name,family,field,data,meta,model,levels,rows,receipt,near_half)
    if "multiscale/report.json" in sources:
        from .directional_probability import probability_plate
        probability_plate(dest,field,model,data,meta,name,receipt)
    receipt["figure_sha256"]=digest(dest/"deposit.png")
    write_json(dest/"deposit.json",receipt)
    page=folder/"README.md"
    old_hash=digest(page) if page.exists() else None
    old = page.read_text() if page.exists() else ""
    body = deposit_page(old,family,name,receipt,report)
    if "multiscale/report.json" in sources and "\n[Previous and revised isopachs at three scales]" in old:
        body = body.rstrip()+"\n"+old[old.index("\n[Previous and revised isopachs at three scales]"):]
    if "footprints/report.json" in sources:
        split = old.find("\n[Footprint model and assumptions]")
        if split < 0:
            split = old.find("\n## Geographic support and extrapolation")
        if split >= 0:
            body = body.rstrip()+"\n"+old[split:]
    (dest/"README.md").write_text(body)
    manifest={"directional":"azimuth/artifacts.json","spherical-mixture":"artifacts.json"}.get(family)
    if manifest and (folder/manifest).is_file():
        hashes=read_json(folder/manifest)
        key="../README.md" if family=="directional" else "README.md"
        if hashes[key]!=old_hash:
            raise ValueError("Published family page differs from its recorded checksum")
        hashes[key]=digest(dest/"README.md")
        (dest/manifest).parent.mkdir(exist_ok=True)
        write_json(dest/manifest,hashes)
    return receipt

def align_near_views(home, data, meta, *, source_home=None):
    """Reframe saved maps on one input after all available peer fits have been rendered."""
    home = Path(home)
    source = Path(source_home or home)
    charts = {}
    for family in FAMILIES:
        path = home/family/"median-grid.npz"
        if path.is_file():
            receipt = read_json(home/family/"deposit.json")
            if receipt["input_sha256"]!=meta["input_sha256"] or digest(path)!=receipt["median_grid_sha256"]:
                raise ValueError("Cannot match near-vent windows across different data or maps")
            with np.load(path,allow_pickle=False) as saved:
                field = PolarField(saved["radii_km"],saved["log10_cm"],float(saved["axis"]))
            charts[family] = field,receipt
    if not charts:
        return
    half = near_frame([f for f,_ in charts.values()],data)
    for family,(field,receipt) in charts.items():
        if receipt["near_half_width_km"]==half:
            continue
        _,model,sources = open_field(source/family,family,data,meta)
        if sources!=receipt["sources_sha256"]:
            raise ValueError("Near-vent window belongs to a different fitted posterior")
        folder=home/family
        rows=np.genfromtxt(folder/"thickness-area.csv",names=True,delimiter=",",dtype=None,encoding="utf-8",ndmin=1)
        receipt.pop("figure_sha256")
        plate(folder/"deposit.png",receipt["eruption"],family,field,data,meta,model,np.array(receipt["levels_cm"]),rows,receipt,half)
        receipt["figure_sha256"]=digest(folder/"deposit.png")
        write_json(folder/"deposit.json",receipt)
