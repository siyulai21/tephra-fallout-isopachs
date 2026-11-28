"""Regression checks for the complete wrapped field, partition and whole-Earth integral."""

from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import norm, t
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.models.wrapped_angular.conditioning import single_event, bounded_normal
from src.empirical_isopach.models.wrapped_angular.geometry import basis, heat_weights
from src.empirical_isopach.models.wrapped_angular.packet import MapView, read, write
from src.empirical_isopach.models.wrapped_angular.fitting import fit, source_for
from src.empirical_isopach.models.wrapped_angular.integrate import integrate
from src.empirical_isopach.models.wrapped_angular.source import student_events
from src.empirical_isopach.models.wrapped_angular.validation import partitions


def four_kinds():
    d, _ = audited_data(resolve_dataset("aso-4"))
    # Use a well-observed flank so the finite geometry library is identified
    # even in the reduced-cost regression test.
    n = len(d.xy)
    xy = d.xy.copy()
    east, north = xy.T
    radius = np.hypot(east, north)
    values = np.log(100.0) - 1.35 * np.log1p(radius / 22.0) - 0.003 * (radius - 0.92 * east)
    values[5:7] = [-1.6, 1.2]
    censored = np.zeros(n, bool)
    censored[5:7] = True
    interval = np.zeros(n, bool)
    interval[6] = True
    minimum = np.zeros(n, bool)
    minimum[1] = True
    return replace(
        d,
        log_value=values,
        log_sd=np.full(n, 0.15),
        censored=censored,
        lower_censored=interval,
        interval_log_upper=np.where(interval, 1.65, np.nan),
        reported_minimum=minimum,
    )


def test_periodic_and_antipodal_covariance():
    p = np.array(
        [[0.0, 0.0], [20.0, -50.0], [0.0, 2.0 * 6371.007180918], [0.000001, 100.0], [-0.000001, 100.0]]
    )
    a = basis(p, 1.0, 3.0, 2.0, 2.0)
    b = basis(p, 1.0, 3.0, 2.0, 2.0, "three-harmonic")
    np.testing.assert_allclose((a * a).sum(1), (b * b).sum(1), rtol=1.0e-13)
    assert not a[[0, 2], 8:].any()
    np.testing.assert_allclose(a[-1], a[-2], atol=1.0e-7)
    w = heat_weights(2.0) ** 2
    ref = np.r_[1.0, np.arange(1, 4, dtype=float) ** -4]
    np.testing.assert_allclose([w.sum(), w @ np.arange(9) ** 2], [ref.sum(), ref @ np.arange(4) ** 2])


def test_event_likelihoods_and_positive_tail_inverse():
    dataset = four_kinds()
    for i in (5, 6):
        d = dataset.subset([i])
        st = single_event(np.array([[0.8]]), np.array([0.6]), d, 0.4)
        sigma = np.sqrt(0.8**2 + 0.4**2 + d.log_sd**2)
        p = norm.cdf((d.log_value - 0.6) / sigma)
        if d.interval[0]:
            p = norm.cdf((d.interval_log_upper - 0.6) / sigma) - p
        np.testing.assert_allclose(st["logz"], np.log(p), rtol=1.0e-11)
    q = bounded_normal(np.array([20.0]), np.array([21.0]), np.array([0.5])).item()
    assert 20 < q < 21
    x = np.array([0.4, 1.0, 2.0])
    hi = np.array([np.nan, np.nan, 2.1])
    np.testing.assert_allclose(
        student_events(x, hi, np.arange(3), 0.2, 0.6),
        [
            t.logpdf((x[0] - 0.2) / 0.6, 4) - np.log(0.6),
            t.logcdf((x[1] - 0.2) / 0.6, 4),
            np.log(t.sf((x[2] - 0.2) / 0.6, 4) - t.sf((hi[2] - 0.2) / 0.6, 4)),
        ],
        rtol=1.0e-10,
    )


def test_four_kind_identity_draws_and_integral(tmp_path):
    d = four_kinds()
    second = replace(d, reported_minimum=np.zeros(len(d.xy), bool))
    a, _, ev, _ = fit(d, 0.0, bound_power=7)
    b, _, ev2, _ = fit(second, 0.0, bound_power=7)
    xy = np.vstack(([0.0, 0.0], d.xy[:4]))
    np.testing.assert_array_equal(ev["log_score"], ev2["log_score"])
    np.testing.assert_array_equal(a.draws(xy), b.draws(xy))
    np.testing.assert_allclose(a.draws(xy), np.concatenate([a.draws(xy[:2]), a.draws(xy[2:])], axis=-1))
    q = a.quantiles(xy)
    assert np.all(np.diff(q, axis=0) >= 0)
    meta = dict(input_sha256="same", vent_lonlat=[0.0, 0.0])
    path = tmp_path / "posterior.npz"
    write(path, a, meta)
    np.testing.assert_allclose(MapView(read(path, meta), 0.0).central(xy), q[1] / np.log(10.0))
    with pytest.raises(ValueError, match="edition"):
        read(path, {**meta, "input_sha256": "different"})
    rules = ((12, 12, 0.5), (14, 16, 0.25))
    va = integrate(a, d.xy, 0.0, rules=rules)
    vb = integrate(b, d.xy, 0.0, rules=rules)
    np.testing.assert_array_equal(va["volumes_km3"], vb["volumes_km3"])
    assert 0 < va["volumes_km3"][0] < va["volumes_km3"][2]


def test_source_sections_cannot_move_exterior():
    d = four_kinds()
    xy = d.xy.copy()
    xy[:2] = [[0.0, 0.0], [0.6, -0.2]]
    d = replace(d, xy=xy)
    source = dict(area_km2=4 * np.pi, centroid_km=[0.0, 0.0], covariance_km2=np.eye(2))
    record = dict(vent_lonlat=[0.0, 0.0], input_sha256="custom")
    assert source_for(record, d) is None
    value = d.log_value.copy()
    value[:2] += np.log(1000.0)
    before, *_ = fit(d, 0.0, source, bound_power=6, refined_source=False)
    after, *_ = fit(replace(d, log_value=value), 0.0, source, bound_power=6, refined_source=False)
    np.testing.assert_array_equal(before.quantiles(xy[2:]), after.quantiles(xy[2:]))
    assert after.quantiles(xy[:1], (0.5,))[0, 0] > before.quantiles(xy[:1], (0.5,))[0, 0]


def test_registered_geometry_is_invariant_to_minimum_provenance():
    data, meta = audited_data(resolve_dataset("mazama"))
    explicit = replace(data, reported_minimum=np.zeros(len(data.xy), bool))
    assert source_for(meta, data) == source_for({**meta, "input_sha256": "equivalent CSV"}, explicit)
    assert source_for({**meta, "input_sha256": "other event"}, explicit.subset([0])) is None
    current = resolve_dataset("b-tm")
    data, meta = audited_data(current)
    for design in ("geographic", "quadrants", "distal"):
        for _, tr, te in partitions(data, meta, current, design):
            if tr is not None:
                assert not np.intersect1d(tr.row_ids, te.row_ids).size


def test_independent_volume_normalization_and_joint_fields():
    from src.probabilistic_isopach.global_volume import EarthDomain
    from src.empirical_isopach.models.wrapped_angular.integrate import joint_volumes

    class Constant:
        source = None
        params = None

        def __init__(self):
            self.regional = self

        def local(self, xy):
            return np.zeros(len(xy), bool)

        def coherent(self, count, seed):
            return lambda xy: np.full((2, count, len(xy)), np.log(7.0))

        def bank(self, xy):
            return self.coherent(4, 1)(xy)

        def quantiles(self, xy, p, **kwargs):
            return np.broadcast_to(np.log([4.0, 7.0, 12.0])[:, None], (3, len(xy)))

    data = four_kinds()
    dep = Constant()
    v = integrate(dep, data.xy, 0.0)
    one_cm = 1.0e-5 * EarthDomain.wgs84(0.0).describe()["surface_area_km2"]
    np.testing.assert_allclose(v["volumes_km3"], one_cm * np.array([4.0, 7.0, 12.0]), rtol=5.0e-6)
    joint = joint_volumes(dep, data, 0.0, counts=(2, 4))
    assert joint["passed"]
    np.testing.assert_allclose(joint["quantiles_km3"], 7 * one_cm, rtol=5.0e-6)


def test_resume_requires_complete_matching_receipts(tmp_path):
    import json
    from src.empirical_isopach.models.wrapped_angular.report import (
        verified_release,
        write_manifest,
        RELEASE_FILES,
    )

    identity = {"edition": "first"}
    for name in RELEASE_FILES:
        (tmp_path / name).write_bytes(b"file")
    write_manifest(tmp_path, identity)
    assert verified_release(tmp_path, identity)
    assert not verified_release(tmp_path, {"edition": "other"})
    (tmp_path / "posterior.npz").write_bytes(b"changed")
    assert not verified_release(tmp_path, identity)
    (tmp_path / "artifacts.json").write_text(json.dumps({"identity": identity, "artifacts_sha256": {}}))
    assert not verified_release(tmp_path, identity)
    (tmp_path / "artifacts.json").write_text("{")
    assert not verified_release(tmp_path, identity)


def test_shared_frame_keeps_the_complete_release_replayable(tmp_path, monkeypatch):
    from pathlib import Path
    from shutil import copytree
    from src.empirical_isopach.core.audit import audited_data
    from src.empirical_isopach.core.catalog import resolve_dataset
    from src.empirical_isopach.models.wrapped_angular.packet import FAMILY
    from src.empirical_isopach.models.wrapped_angular.report import verified_release
    from src.empirical_isopach.visualization import deposit

    data, meta = audited_data(resolve_dataset("tambora"))
    folder = tmp_path / FAMILY
    copytree(Path(__file__).resolve().parents[1] / "eruptions/tambora/results" / FAMILY, folder)
    original = deposit.read_json(folder / "artifacts.json")
    receipt = deposit.read_json(folder / "deposit.json")
    half = 2 * receipt["near_half_width_km"]
    monkeypatch.setattr(deposit, "near_frame", lambda fields, sites: half)
    deposit.align_near_views(tmp_path, data, meta)
    assert verified_release(folder, original["identity"])
    changed = deposit.read_json(folder / "artifacts.json")["artifacts_sha256"]
    assert {k for k, v in changed.items() if v != original["artifacts_sha256"][k]} == {
        "deposit.json", "deposit.png", "three-scales.png"
    }
    assert deposit.read_json(folder / "deposit.json")["near_half_width_km"] == half
