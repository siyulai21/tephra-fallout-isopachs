"""Publish one global median deposit, its physical integral and its thickness–area curve."""
import csv
import json
from pathlib import Path

import numpy as np

from .deposit_field import FAMILIES, PolarField, digest, display_name, doubling_levels, open_field, review_path, sample_field
from .deposit_integrals import independent_integral, within_cells
from .deposit_maps import LAND, frames, plate
from .maps import boundary_points


def volume_interval(family, report, extent=None):
    """Return only the two conditional volume uncertainty endpoints."""
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
    lo,hi = receipt["conditional_volume_interval95_km3"]
    warning = "" if receipt["complete_field_checks_passed"] else " **Numerical warning on the fitted deposit distribution.**"
    volume=receipt["median_map_volume_km3"]
    volume_label=f"{volume:,.0f}" if volume>=1000 else f"{volume:,.3g}"
    label = display_name(family, receipt["sources_sha256"])
    first = [f"# {name} · {label}", "",
             f"The principal bulk volume is **{volume_label} km³**, the whole-Earth integral of the median-thickness map. "
             f"The conditional 95% interval for total fallout volume is **{lo:,.3g}–{hi:,.3g} km³** across complete posterior draws.{warning}", "",
             f"![{name}, {label}. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)", "",
             "Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. "
             "The volume also includes thinner fallout beyond the 0.5 cm line. "
             "[Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)", ""]
    if not receipt["integration_passed"]:
        first.extend(["**Map integration warning.** The independent or refinement check exceeded its tolerance. See the numerical record.", ""])
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


def describe_surface(field, levels, checks, bounds):
    volume,area=within_cells(field,levels)
    lower_order,_=within_cells(field,order=2,angles=2)
    coarse=PolarField(field.radii[::2],field.log10_cm[::2,::2],field.axis)
    coarse_volume,coarse_area=within_cells(coarse,levels)
    from scipy.interpolate import RegularGridInterpolator
    edge=boundary_points(bounds)
    ex,ey=edge[:,0]/field.axis,edge[:,1]*field.axis
    interp=RegularGridInterpolator((field.radii,np.r_[field.angles,2*np.pi]),
                                  np.column_stack((field.log10_cm,field.log10_cm[:,0])),bounds_error=False,fill_value=np.nan)
    edge_cm=10.**interp((np.hypot(ex,ey),np.arctan2(ex,ey)%(2*np.pi)))
    edge_max=float(np.nanmax(edge_cm)) if np.isfinite(edge_cm).any() else None
    if edge_max is not None and edge_max>=.5:
        raise ArithmeticError("The full-deposit map clips a 0.5 cm contour")
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


def render_deposit(folder, family, data, meta, name, *, output=None, sampled=None):
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
        field,checks=checked_field(view,data,latitude)
    else:
        field,checks,identity=sampled
        if identity!=sources:
            raise ValueError("Scratch field belongs to different saved fits")
    report=read_json(folder/next(iter(sources)))
    interval,fit_ok=volume_interval(family,report,read_json(review) if review.exists() else None)
    maximum=max(field.log10_cm.max(),data.log_value.max()/np.log(10.))
    levels=doubling_levels(10.**maximum)
    bounds,_,_=frames(field,data)
    volume,area,coarse_area,checks_record=describe_surface(field,levels,checks,bounds)
    receipt=dict(protocol="median-thickness-deposit-v1",family=family,eruption=name,domain="WGS84 whole Earth",
                 input_sha256=meta["input_sha256"],sources_sha256=sources,land_sha256=digest(LAND),
                 median_map_volume_km3=volume,conditional_volume_interval95_km3=list(interval),
                 complete_field_checks_passed=bool(fit_ok),**checks_record)
    rows=[dict(thickness_cm=cm,area_km2=a,sqrt_area_km=np.sqrt(a),coarse_area_km2=b,
               resolution_status=("no sampled exceedance" if a==0 else "grid sensitive" if abs(a-b)>max(1.,.05*a) else "resolved"))
          for cm,a,b in zip(levels,area,coarse_area,strict=True)]
    with (dest/"thickness-area.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    receipt["thickness_area_sha256"]=digest(dest/"thickness-area.csv")
    plate(dest/"deposit.png",name,family,field,data,meta,model,levels,rows,receipt)
    receipt["figure_sha256"]=digest(dest/"deposit.png")
    write_json(dest/"deposit.json",receipt)
    page=folder/"README.md"
    old_hash=digest(page)
    (dest/"README.md").write_text(deposit_page(page.read_text(),family,name,receipt,report))
    manifest={"directional":"azimuth/artifacts.json","spherical-mixture":"artifacts.json"}.get(family)
    if manifest:
        hashes=read_json(folder/manifest)
        key="../README.md" if family=="directional" else "README.md"
        if hashes[key]!=old_hash:
            raise ValueError("Published family page differs from its recorded checksum")
        hashes[key]=digest(dest/"README.md")
        (dest/manifest).parent.mkdir(exist_ok=True)
        write_json(dest/manifest,hashes)
    return receipt
