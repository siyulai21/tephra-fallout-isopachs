"""Replay published map identities, volumes, observations and footprint-area curves."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .nested_reach import case_home
from .nested_reach_challenge import records
from .principal_volumes import catalogue_rows
from .refresh_figures import recorded_observations, regional_models
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.visualization.deposit import FAMILIES, volume_interval, digest, open_field, read_json
from src.empirical_isopach.visualization.deposit_maps import LAND, local_half_width, outer_contour
from src.empirical_isopach.visualization.deposit_field import PolarField, polar_nodes
from src.empirical_isopach.visualization.deposit_integrals import within_cells
from src.empirical_isopach.visualization.contour_limits import inner_contour, near_frame, near_levels, visible_levels
from src.empirical_isopach.visualization.shorelines import detail
from src.empirical_isopach.visualization.thickness_band import field_quantiles, regional_band
from src.empirical_isopach.visualization.principal import case_section


def verify_global(folder, family, data, meta):
    import json
    r=read_json(folder/"deposit.json")
    legacy = r["protocol"]=="median-thickness-deposit-v1"
    assert (legacy or r["protocol"]=="median-thickness-deposit-v2") and r["family"]==family
    assert r["input_sha256"]==meta["input_sha256"] and r["land_sha256"]==digest(LAND)
    assert r["figure_sha256"]==digest(folder/"deposit.png") and r["thickness_area_sha256"]==digest(folder/"thickness-area.csv")
    with Image.open(folder/"deposit.png") as image:
        assert json.loads(image.info["Description"])=={k:v for k,v in r.items() if k!="figure_sha256"}
    median,model,src=open_field(folder,family,data,meta)
    assert src==r["sources_sha256"]
    fit=read_json(folder/next(iter(src)))
    if "multiscale/report.json" in src:
        assert fit["independent_draw_integration"] and fit["fit"]["retained_mass"]>=.9995-1e-10
        assert digest(folder/"multiscale/posterior.npz")==fit["artifacts_sha256"]["posterior.npz"]
    elif family == "directional":
        assert "median" not in fit["volume"]["coherent_volume_km3"]
    elif family == "spherical-mixture":
        assert all({"median","mean"}.isdisjoint(item["volume"]) for item in fit["fits"].values())
    else:
        assert {"volume_km3","conditional_volume_km3"}.isdisjoint(fit)
    revised = folder/"extent/report.json"
    q,ok=volume_interval(family,fit,read_json(revised) if revised.exists() else None)
    outer = .5 if legacy else r["outer_contour"]["level_cm"]
    if legacy:
        np.testing.assert_allclose(q,r["conditional_volume_interval95_km3"])
        assert ok==r["complete_field_checks_passed"]
    else:
        assert ok==r["posterior_checks_passed"]
        import json
        path = folder/"median-grid.npz"
        assert r["median_grid_sha256"]==digest(path)
        with np.load(path,allow_pickle=False) as packet:
            assert str(packet["input_sha256"])==meta["input_sha256"]
            assert json.loads(str(packet["sources_json"]))==r["sources_sha256"]
            grid = PolarField(packet["radii_km"],packet["log10_cm"],float(packet["axis"]))
        replayed,areas = within_cells(grid,r["levels_cm"])
        np.testing.assert_allclose(replayed,r["median_map_volume_km3"],rtol=1e-12)
        np.testing.assert_allclose(areas,pd.read_csv(folder/"thickness-area.csv").area_km2,rtol=1e-10,atol=1e-10)
        border = outer_contour(grid,data)
        for key in border:
            np.testing.assert_allclose(border[key],r["outer_contour"][key],rtol=1e-11,atol=1e-10)
        inner = inner_contour(border)
        assert inner==r["inner_contour"] and inner["inside"]>=1 and inner["next_level_inside"]==0
        h=r["near_half_width_km"]
        assert r["near_visible_levels_cm"]==visible_levels(grid,h,near_levels(grid,inner["level_cm"]))
        assert r["near_shoreline"]==detail(np.array(r["near_bounds_km"]))
        verify_band_and_extent(r,median,data,meta)
    areas=pd.read_csv(folder/"thickness-area.csv")
    levels=areas.thickness_cm.to_numpy()
    np.testing.assert_array_equal(levels,.5*2.**np.arange(len(levels)))
    np.testing.assert_array_equal(levels,r["levels_cm"])
    volume=r["median_map_volume_km3"]
    points=np.array(r["check_points_x_y_km_log10_cm"])
    np.testing.assert_allclose(median.central(points[:,:2]),points[:,2],atol=1e-9)
    area=areas["area_km2"].to_numpy()
    np.testing.assert_allclose(area,areas["sqrt_area_km"]**2,rtol=1e-12)
    assert (area>=0).all() and (np.diff(area)<=0).all() and area[0]<=median.domain.describe()["surface_area_km2"]
    edge=r["maximum_display_edge_cm"]
    assert edge is None or edge<outer
    trials=r["independent_integrals"][-1]
    error=abs(volume/np.mean(trials["km3"])-1)
    np.testing.assert_allclose(error,r["field_replay_relative_difference"])
    passed=(error<.015 and abs(r["coarse_grid_volume_km3"]/volume-1)<.02 and
            abs(r["lower_quadrature_order_volume_km3"]/volume-1)<.001 and trials["relative_change"]<.01)
    assert passed==r["integration_passed"]
    return r["near_bounds_km"]


def verify_band_and_extent(r,view,data,meta):
    band = r["thickness_band"]
    np.testing.assert_array_equal(band["probabilities"],[.25,.5,.75])
    probes = np.array(band["check_points_x_y_km_log10_q25_q50_q75_cm"])
    np.testing.assert_allclose(field_quantiles(view,probes[:,:2]).T,probes[:,2:],atol=1e-8)
    volumes = np.array(band["volumes_km3"])
    assert 0 < volumes[0] <= volumes[1] <= volumes[2]
    np.testing.assert_allclose(volumes[1],r["median_map_volume_km3"],rtol=1e-12)
    t = band["independent_integrals"][-1]
    relative = abs(volumes/np.mean(t["km3"],axis=0)-1.)
    np.testing.assert_allclose(relative,band["relative_difference"])
    assert band["passed"]==bool(max(relative)<.015 and max(abs(band["coarse_volumes_km3"]/volumes-1))<.02 and max(t["relative_change"])<.01)
    # Evaluate the four original posterior grid nodes around every accepted site.
    radii,_,axis = polar_nodes(data,meta["vent_lonlat"][1],*r["grid_shape"])
    xy = data.xy
    n = r["grid_shape"][1]
    radius = np.hypot(xy[:,0]/axis,xy[:,1]*axis)
    bearing = np.arctan2(xy[:,0]/axis,xy[:,1]*axis)%(2*np.pi)
    i = np.clip(np.searchsorted(radii,radius,side="right")-1,0,len(radii)-2)
    angular = bearing*n/(2*np.pi)
    j = np.floor(angular).astype(int)
    wr,wa = (radius-radii[i])/(radii[i+1]-radii[i]),angular-j
    mapped = np.zeros(len(xy))
    for di,radial in enumerate((1.-wr,wr)):
        for dj,angular_weight in enumerate((1.-wa,wa)):
            a = 2*np.pi*((j+dj)%n)/n
            rr = radii[i+di].copy()
            rr[i+di==len(radii)-1] *= 1.-1e-8
            original = view.central(np.column_stack((rr*np.sin(a)*axis,rr*np.cos(a)/axis)))
            mapped += radial*angular_weight*original
    border = r["outer_contour"]
    np.testing.assert_array_equal(border["row_ids"],data.row_ids)
    np.testing.assert_allclose(mapped,border["mapped_log10_cm"],rtol=1e-7,atol=1e-7)
    level = border["level_cm"]
    count = lambda cm:int((mapped>=np.log10(cm)-1e-12).sum())
    assert level>=1 and np.log2(level).is_integer()
    assert len(xy)==border["accepted_rows"] and count(level)==border["inside"]
    required = int(np.ceil(.95*len(xy)))
    assert required==border["minimum_enclosed"] and count(2*level)==border["next_level_inside"]<required
    assert border["target_met"]==(count(level)>=required) and (level==1 or border["target_met"])
    np.testing.assert_array_equal(r["full_levels_cm"],[v for v in r["levels_cm"] if level<=v<=r["inner_contour"]["level_cm"]])
    half = r["near_half_width_km"]
    assert half>=local_half_width(data)
    np.testing.assert_array_equal(r["near_bounds_km"],[-half,-half,half,half])
    assert r["near_sites"]==int((np.abs(xy)<=half).all(axis=1).sum())


def verify_case(record):
    home=case_home(record["slug"])
    preview=record["slug"]=="pinatubo-1991"
    region=home/"regional" if preview else home
    data,meta=audited_data(record["path"],allow_intervals=True)
    frames=[verify_global(home/key,key,data,meta) for key in FAMILIES]
    # The two archived fits and the current nested fit may use different PROJ builds.
    for near_bounds in frames[1:]:
        np.testing.assert_allclose(near_bounds,frames[0],rtol=0.,atol=1e-9)
    if not preview:
        charts = []
        for key in FAMILIES:
            with np.load(home/key/"median-grid.npz",allow_pickle=False) as packet:
                charts.append(PolarField(packet["radii_km"],packet["log10_cm"],float(packet["axis"])))
        np.testing.assert_allclose(frames[0][2],near_frame(charts,data))
    verify_regional(record, region, meta, check_bands=not preview)


def verify_regional(record, region, meta, *, check_bands=True):
    """Replay current and historical median maps from the fitted regional parameters."""
    r=read_json(region/"report.json")
    for item in r["methods"].values():
        assert "expected_km3" not in item["regional_volume"]
        assert "0.5" not in item["joint_volume"]["conditional_quantiles_km3"]
    recorded, _ = recorded_observations(record, region)
    models,hull,_=regional_models(recorded,r)
    screen=read_json(region/"map-display.json")
    bounds=np.array(screen["display_bounds_km"])
    assert screen["land_sha256"]==digest(LAND) and screen["input_sha256"]==meta["input_sha256"]
    assert ((hull.vertices>=bounds[:2]) & (hull.vertices<=bounds[2:])).all()
    for k,fit in models.items():
        if check_bands:
            b = read_json(region/"thickness-bands.json")
            assert b["input_sha256"]==meta["input_sha256"] and b["regional_report_sha256"]==digest(region/"report.json")
            replay = regional_band(fit,hull)
            np.testing.assert_allclose(replay["volumes_km3"],b["methods"][k]["volumes_km3"],rtol=3e-6)
            assert replay["passed"]==b["methods"][k]["passed"]
        probes=np.array(screen["check_points_x_y_km_log10_cm"][k])
        predicted = predictive_distribution(fit, probes[:, :2]).quantile(.5)
        # One part per million in thickness at every site, including sub-centimetre ash.
        np.testing.assert_allclose(predicted, probes[:, 2] * np.log(10.),
                                   rtol=0, atol=np.log1p(1e-6), err_msg=f"{record['slug']} {k} median-map probes")


def main():
    for r in records(published=True):
        verify_case(r)
    from .publication import published_methods, result_home
    from src.empirical_isopach.core.catalog import datasets
    for record in datasets():
        home = result_home(record)
        assert case_section(home,"results/") in (home.parent/"README.md").read_text()
        if "published_methods" in record:
            data, meta = audited_data(record["path"])
            if {"stacked", "spline"} <= set(published_methods(record)):
                verify_regional(record, home, meta)
            frames = [verify_global(home/key,key,data,meta) for key in FAMILIES if key in published_methods(record)]
            for frame in frames[1:]:
                np.testing.assert_allclose(frame,frames[0],rtol=0.,atol=1e-9)
    expected=pd.DataFrame(catalogue_rows())
    pd.testing.assert_frame_equal(pd.read_csv("comparisons/principal-volumes.csv").convert_dtypes(),expected.convert_dtypes(),check_dtype=False)
    print(f"Replayed the median global field for every family and the two regional fits across {len(records(published=True))} case records.")


if __name__=="__main__":
    main()
