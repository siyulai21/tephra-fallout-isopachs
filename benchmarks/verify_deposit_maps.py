"""Replay published map identities, volumes, observations and footprint-area curves."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .nested_reach import case_home
from .nested_reach_challenge import records
from .principal_volumes import catalogue_rows
from .refresh_figures import regional_models
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.distributions import predictive_distribution
from src.empirical_isopach.visualization.deposit import FAMILIES, volume_interval, digest, open_field, read_json
from src.empirical_isopach.visualization.deposit_maps import LAND
from src.empirical_isopach.visualization.principal import case_section


def verify_global(folder, family, data, meta):
    import json
    r=read_json(folder/"deposit.json")
    assert r["protocol"]=="median-thickness-deposit-v1" and r["family"]==family
    assert r["input_sha256"]==meta["input_sha256"] and r["land_sha256"]==digest(LAND)
    assert r["figure_sha256"]==digest(folder/"deposit.png") and r["thickness_area_sha256"]==digest(folder/"thickness-area.csv")
    with Image.open(folder/"deposit.png") as image:
        assert json.loads(image.info["Description"])=={k:v for k,v in r.items() if k!="figure_sha256"}
    median,model,src=open_field(folder,family,data,meta)
    assert src==r["sources_sha256"]
    fit=read_json(folder/next(iter(src)))
    if family == "directional":
        assert "median" not in fit["volume"]["coherent_volume_km3"]
    elif family == "spherical-mixture":
        assert all({"median","mean"}.isdisjoint(item["volume"]) for item in fit["fits"].values())
    else:
        assert {"volume_km3","conditional_volume_km3"}.isdisjoint(fit)
    q,ok=volume_interval(family,fit)
    np.testing.assert_allclose(q,r["conditional_volume_interval95_km3"])
    assert ok==r["complete_field_checks_passed"]
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
    assert edge is None or edge<.5
    trials=r["independent_integrals"][-1]
    error=abs(volume/np.mean(trials["km3"])-1)
    np.testing.assert_allclose(error,r["field_replay_relative_difference"])
    passed=(error<.015 and abs(r["coarse_grid_volume_km3"]/volume-1)<.02 and
            abs(r["lower_quadrature_order_volume_km3"]/volume-1)<.001 and trials["relative_change"]<.01)
    assert passed==r["integration_passed"]
    return r["near_bounds_km"]


def verify_case(record):
    home=case_home(record["slug"])
    preview=record["slug"]=="pinatubo-1991"
    region=home/"regional" if preview else home
    data,meta=audited_data(record["path"],allow_intervals=True)
    frames=[verify_global(home/key,key,data,meta) for key in FAMILIES]
    assert frames[0]==frames[1]==frames[2]
    published=(home if preview else home.parent)/"README.md"
    assert case_section(home,"" if preview else "results/",regional_subdir="regional" if preview else "") in published.read_text()
    r=read_json(region/"report.json")
    for item in r["methods"].values():
        assert "expected_km3" not in item["regional_volume"]
        assert "0.5" not in item["joint_volume"]["conditional_quantiles_km3"]
    models,hull,_=regional_models(data,r)
    screen=read_json(region/"map-display.json")
    bounds=np.array(screen["display_bounds_km"])
    assert screen["land_sha256"]==digest(LAND) and screen["input_sha256"]==meta["input_sha256"]
    assert ((hull.vertices>=bounds[:2]) & (hull.vertices<=bounds[2:])).all()
    for k,fit in models.items():
        probes=np.array(screen["check_points_x_y_km_log10_cm"][k])
        np.testing.assert_allclose(predictive_distribution(fit,probes[:,:2]).quantile(.5)/np.log(10.),probes[:,2],atol=2e-7)


def main():
    for r in records():
        verify_case(r)
    expected=pd.DataFrame(catalogue_rows())
    pd.testing.assert_frame_equal(pd.read_csv("comparisons/principal-volumes.csv").convert_dtypes(),expected.convert_dtypes(),check_dtype=False)
    print(f"Replayed the median global field for every family and the two regional fits across {len(records())} case records.")


if __name__=="__main__":
    main()
