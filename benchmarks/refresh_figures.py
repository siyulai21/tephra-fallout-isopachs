"""Replay the matched five-family study in fresh figures. Current global plates use benchmarks.deposit_maps."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .publication import published_datasets as datasets, result_home
from src.empirical_isopach.visualization.comparison import draw_comparison, saved_observations, saved_regional_grid
from src.empirical_isopach.visualization.regional import regional_grid
from src.empirical_isopach.models.mass_mixture.study import audited_data
from src.empirical_isopach.models.gaussian_process.model import SpatialModel, Candidate
from src.empirical_isopach.models.gaussian_process.ensemble import StackedModel
from src.empirical_isopach.models.cubic_spline.model import TrendRegularizedSpline, SplineCandidate
from src.empirical_isopach.region import SurveyRegion, integrate_region
from src.empirical_isopach.reporting import save_figures
from src.empirical_isopach.global_reporting import figures as spherical_figures
from src.empirical_isopach.models.directional.reporting import figures as directional_figures
from src.empirical_isopach.models.directional.geometry import Geometry
from src.empirical_isopach.support import SupportMesh, POLICIES
from src.empirical_isopach.support_reporting import support_figures
from src.empirical_isopach.visualization.summary import case_overview, all_case_scores, principal_report
from src.empirical_isopach.visualization.maps import grid_axes
from src.empirical_isopach.core.distributions import predictive_distribution


ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(Path(p).read_text())
digest = lambda p: sha256(Path(p).read_bytes()).hexdigest()


def recorded_observations(record, home):
    current, meta = audited_data(record["path"])
    data = saved_observations(home/"accepted_observations.csv")
    by_id = {v:i for i,v in enumerate(current.row_ids)}
    current = current.subset([by_id[i] for i in data.row_ids])
    np.testing.assert_allclose(current.interval_log_upper, data.interval_log_upper, equal_nan=True)
    np.testing.assert_allclose(current.xy, data.xy, rtol=1e-12, atol=1e-7)
    for key in ("censored", "lower_censored", "reported_minimum", "row_ids"):
        np.testing.assert_array_equal(getattr(current, key), getattr(data, key))
    for key in ("log_value", "log_sd"):
        np.testing.assert_allclose(getattr(current, key), getattr(data, key), rtol=1e-13, atol=1e-14)
    return data, meta


def same_description(actual, expected, where="model"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), where
        for key in expected:
            same_description(actual[key], expected[key], where+"."+key)
    elif where.rsplit(".", 1)[-1] in {"beta", "trend_beta"}:
        # On a ray u, the fitted slope is -beta[1] + beta[2:] @ u.
        # Its maximum absolute change over every unit bearing is exact, and
        # does not depend on the map axes or a component being close to zero.
        fitted, reference = np.asarray(actual, float), np.asarray(expected, float)
        assert fitted.shape == reference.shape == (4,), where
        same_description(fitted[0], reference[0], where + ".intercept")
        difference = fitted - reference
        slope_error = abs(difference[1]) + np.linalg.norm(difference[2:])
        slope_scale = abs(reference[1]) + np.linalg.norm(reference[2:])
        assert slope_error <= 1e-9 + 2e-7 * slope_scale, (where, slope_error, slope_scale)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for i, (a, b) in enumerate(zip(actual, expected)):
            same_description(a, b, where+f"[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(actual, expected, rtol=2e-7, atol=1e-9), (where, actual, expected)
    else:
        assert actual == expected, (where, actual, expected)


def regional_fields(data, report):
    """Refit the saved regional specifications on supplied observation arrays."""
    fitted = {}
    gp = report["methods"]["stacked"]["model"]
    components = [SpatialModel(Candidate(**c["model"]["candidate"])).fit(data) for c in gp["components"]]
    fitted["stacked"] = StackedModel(components, [c["weight"] for c in gp["components"]],
                                     gp["observation_discrepancy_variance_log"])
    spline = report["methods"]["spline"]["model"]
    fitted["spline"] = TrendRegularizedSpline(SplineCandidate(**spline["candidate"]),
        quadrature_order=spline["quadrature_order"]).fit(data, bounds=report["fit_bounds_km"])
    return fitted


def regional_models(data, report):
    fitted = regional_fields(data, report)
    checks = {}
    region = SurveyRegion(data.xy)
    for key, model in fitted.items():
        same_description(model.describe(), report["methods"][key]["model"], key)
        integral = integrate_region(model, region)
        old = report["methods"][key]["regional_volume"]
        expected = old["refinement"][-1]["expected_km3"]
        np.testing.assert_allclose(integral["expected_km3"], expected, rtol=2e-7, atol=1e-9)
        np.testing.assert_allclose(integral["median_surface_km3"], old["median_surface_km3"], rtol=2e-7, atol=1e-9)
        checks[key] = {"recorded_parameters_replayed":True, "selection_repeated":False,
            "expected_volume_relative_difference":abs(integral["expected_km3"]-expected)/expected,
            "median_surface_volume_relative_difference":abs(integral["median_surface_km3"]-old["median_surface_km3"])/old["median_surface_km3"]}
    return fitted, region, checks


def refresh_case(record, output, scopes=("regional", "directional", "spherical", "comparison")):
    slug, name = record["slug"], record["name"]
    home = ROOT/result_home(record)
    target = Path(output)/slug
    target.mkdir(parents=True, exist_ok=True)
    data, meta = recorded_observations(record, home)
    regional = read(home/"report.json")
    directional = read(home/"directional/single-structure/report.json")
    spherical = read(home/"spherical-mixture/report.json")
    assert all(r["input"]["input_sha256"] == meta["input_sha256"] for r in (regional, directional, spherical))
    checks = {"input_sha256":meta["input_sha256"], "source_reports":{
        key:digest(home/key) for key in ("report.json", "directional/single-structure/report.json", "spherical-mixture/report.json")}}
    if "regional" in scopes or "comparison" in scopes:
        print(f"{name}: reconstructing recorded regional fits (no selection)", flush=True)
        models, region, parity = regional_models(data, regional)
        checks["regional_parity"] = parity
    if "regional" in scopes:
        save_figures(target, name, data, meta, region, models, regional)
        support = target/"support"
        support.mkdir(exist_ok=True)
        saved = read(home/"support/report.json")
        meshes = {p.name:SupportMesh(data, p) for p in POLICIES}
        x, y = grid_axes(np.array(regional["fit_bounds_km"]), longest=257)
        xx, yy = np.meshgrid(x, y)
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        maps = {}
        for key, model in models.items():
            # No case has eligible contours. Keep the stored decisions; never
            # promote a newly drawn contour to an audited one during replay.
            assert not any(r["policies"]["standard"]["eligible"] for r in saved["methods"][key]["contours"]["levels"])
            z = predictive_distribution(model, xy).quantile(.5).reshape(xx.shape)
            maps[key] = ((x, y, z), {})
        support_figures(support, name, data, meshes, maps, saved, vent=meta["vent_lonlat"])
    if "spherical" in scopes:
        print(f"{name}: rendering saved spherical posterior", flush=True)
        dest = target/"spherical-mixture"
        dest.mkdir(exist_ok=True)
        shutil.copyfile(home/"spherical-mixture/posterior.npz", dest/"posterior.npz")
        with np.load(dest/"posterior.npz") as f:
            samples = dict(f)
        from src.empirical_isopach.models.mass_mixture.bearings.field import PROTOCOL
        from src.empirical_isopach.models.mass_mixture.bearings.maps import BearingView
        bearer = spherical.get("protocol")==PROTOCOL
        if bearer:
            (dest/"bearing").mkdir(exist_ok=True)
            shutil.copyfile(home/"spherical-mixture/bearing/baseline.npz",dest/"bearing/baseline.npz")
        spherical_figures(dest, name, data, meta, samples, spherical)
        if (home/"spherical-mixture/source.png").exists():
            from src.empirical_isopach.source_reporting import source_figure
            source_figure(dest, name, data, meta, samples,view=BearingView(samples,data,meta["vent_lonlat"][1],spherical["bearing"]["map_draw_check"]["draws_per_chain"]) if bearer else None)
        checks["spherical_posterior_sha256"] = digest(dest/"posterior.npz")
    if "directional" in scopes:
        print(f"{name}: rendering saved directional posterior", flush=True)
        dest = target/"directional/single-structure"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(home/"directional/single-structure/posterior.npz", dest/"posterior.npz")
        with np.load(dest/"posterior.npz") as f:
            draws = dict(f)
        g = Geometry(data, meta["vent_lonlat"][1])
        arcs = pd.read_csv(home/"directional/single-structure/isopach-fragments.csv")
        directional_figures(dest, name, data, meta, g, draws, arcs, directional["fits"]["reference"])
        checks["directional_posterior_sha256"] = digest(dest/"posterior.npz")
    ensemble_dir = home/"directional"/"ensemble"
    if (ensemble_dir/"report.json").is_file():
        directional = read(ensemble_dir/"report.json")
        assert directional["input"]["input_sha256"] == meta["input_sha256"]
        checks["source_reports"]["directional/ensemble/report.json"] = digest(ensemble_dir/"report.json")
        if "directional" in scopes:
            from src.empirical_isopach.models.directional.ensemble.fields import load_packet
            from src.empirical_isopach.models.directional.ensemble.maps import render_maps
            ensemble_dest = target/"directional"/"ensemble"
            ensemble_dest.mkdir(parents=True,exist_ok=True)
            fields = load_packet(ensemble_dir,data,meta["vent_lonlat"][1],directional)
            render_maps(ensemble_dest,name,data,meta,fields)
            checks["ensemble_fields_sha256"] = digest(ensemble_dir/"map-fields.npz")
    primary_dir = home/"directional/azimuth"
    if (primary_dir/"report.json").is_file():
        directional = read(primary_dir/"report.json")
        assert directional["input"]["input_sha256"] == meta["input_sha256"]
        checks["source_reports"]["directional/azimuth/report.json"] = digest(primary_dir/"report.json")
        checks["principal_directional_fields_sha256"] = digest(primary_dir/"fields.npz")
        if "directional" in scopes:
            from src.empirical_isopach.models.directional.azimuth.primary import read_field, render_primary
            from src.empirical_isopach.models.directional.archive_readings import load_inventory
            dest = target/"directional/azimuth"
            dest.mkdir(parents=True,exist_ok=True)
            field = read_field(primary_dir,data,directional)
            inventory,_ = load_inventory(record["path"])
            render_primary(dest,data,meta,directional,field,inventory,source=primary_dir)
    if "comparison" in scopes and not (home/"nested-isopachs/deposit.json").is_file():
        if "regional" in scopes:
            comparison_grid = saved_regional_grid(target/"predictions.csv")
        else:
            gx, gy, _, fields, _ = regional_grid(regional["fit_bounds_km"], region, models)
            comparison_grid = gx, gy, fields
        draw_comparison(home, target, comparison_grid)
    if "comparison" in scopes and (home/"nested-isopachs/deposit.json").is_file():
        from src.empirical_isopach.visualization.deposit import FAMILIES, render_deposit
        from src.empirical_isopach.visualization.regional import regional_maps
        if "regional" not in scopes:
            regional_maps(target,name,data,meta,region,models,regional)
        for family in FAMILIES:
            if (home/family/"deposit.json").is_file():
                render_deposit(home/family,family,data,meta,name,output=target/family)
    nested = principal_report(home)
    case_overview(target, name, regional, directional, spherical, nested,map_home=home)
    (target/"figure-replay.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(f"{name}: figures complete", flush=True)
    return name, regional, directional, spherical, nested


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--case", help="Optional accepted-case slug")
    p.add_argument("--scopes", nargs="+", default=["regional", "directional", "spherical", "comparison"],
                   choices=["regional", "directional", "spherical", "comparison"])
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a fresh scratch destination")
    rows = [r for r in datasets() if args.case is None or r["slug"] == args.case]
    if not rows:
        raise ValueError("Unknown case slug")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [refresh_case(r, args.output, args.scopes) for r in rows]
    if len(cases) == len(datasets()):
        all_case_scores(args.output, cases,homes={r["name"]:result_home(r) for r in rows})


if __name__ == "__main__":
    main()
