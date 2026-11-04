"""Independent geometry, analytic integrals and adversarial contour checks."""
import numpy as np
import pytest
from shapely import Polygon, Point, covers, points

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.region import integrate_region
from src.empirical_isopach.support import SupportMesh, SupportPolicy, TriangleRegion
from src.empirical_isopach.contour_support import (
    level_polygons, screen_contours, contiguous_bands, layer_cake_bounds,
    continuation_scenarios, candidate_levels, median_grid)


def observations(xy, values=None):
    xy = np.asarray(xy, float)
    y = 4-.3*np.linalg.norm(xy, axis=1) if values is None else np.asarray(values)
    return Observations(xy, y, np.zeros(len(xy), bool), np.zeros(len(xy)), np.arange(len(xy)))


def lattice():
    x, y = np.meshgrid(np.linspace(-8, 8, 33), np.linspace(-8, 8, 33))
    return np.column_stack((x.ravel(), y.ravel()))


class Radial:
    def __init__(self, shift=0):
        self.shift = shift

    def predict_log(self, xy):
        return 4-.3*np.linalg.norm(xy-np.array([self.shift, 0]), axis=1), np.zeros(len(xy))


def test_support_is_response_duplicate_order_and_unit_invariant():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-10, 10, (80, 2))
    data = observations(xy)
    base = SupportMesh(data)
    changed = observations(np.repeat(xy[::-1], 2, axis=0), np.zeros(160))
    other = SupportMesh(changed)
    assert base.geometry.equals(other.geometry)
    scaled = SupportMesh(observations(100*xy))
    assert scaled.area == pytest.approx(10000*base.area)
    assert np.array_equal(scaled.accepted, base.accepted)


def test_local_mesh_removes_missing_sector_and_internal_hole():
    xy = lattice()
    xy = xy[(np.linalg.norm(xy, axis=1) >= 3) & ~((xy[:, 0] > 0) & (xy[:, 1] > 0))]
    mesh = SupportMesh(observations(xy))
    assert not mesh.geometry.covers(Point(0, 0))
    assert not mesh.geometry.covers(Point(5, 5))
    assert mesh.geometry.covers(Point(-5, -5))


def test_bounds_do_not_fill_sampling_holes():
    xy = lattice()
    data = observations(xy)
    data.censored = np.linalg.norm(xy, axis=1) < 3
    data.lower_censored = data.censored.copy()
    mesh = SupportMesh(data)
    assert not mesh.geometry.covers(Point(0, 0))


def test_constrained_triangulation_preserves_holes_and_integral():
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]])
    region = TriangleRegion.from_polygon(polygon)
    xy, weights = region.quadrature(4)
    assert weights.sum() == pytest.approx(12)
    assert np.all(weights > 0)
    assert np.all(covers(polygon, points(xy)))
    class Constant:
        def predict_log(self, xy):
            return np.full(len(xy), np.log(7)), np.zeros(len(xy))
    volume = integrate_region(Constant(), region)
    assert volume["converged"]
    assert volume["expected_km3"] == pytest.approx(7*12*1e-5)


def test_complete_contours_require_support_and_deletion_stability():
    xy = lattice()
    data = observations(xy)
    mesh = SupportMesh(data)
    grid = median_grid(Radial(), [-8, -8, 8, 8], 129)
    levels = np.exp([2.5, 2.8, 3.1])
    rows, _ = screen_contours(data, grid, [grid]*4, {"standard": mesh}, levels)
    assert all(r["policies"]["standard"]["eligible"] for r in rows)
    shifted = median_grid(Radial(3), [-8, -8, 8, 8], 129)
    failed, _ = screen_contours(data, grid, [shifted]*4, {"standard": mesh}, levels)
    assert not any(r["policies"]["standard"]["eligible"] for r in failed)
    missing = observations(xy[~((xy[:, 0] > 0) & (xy[:, 1] > 0))])
    failed, _ = screen_contours(missing, grid, [grid]*4, {"standard": SupportMesh(missing)}, levels)
    assert all("perimeter_leaves_local_support" in r["policies"]["standard"]["reasons"] for r in failed)


def test_open_contours_are_not_artificially_closed_at_window():
    grid = median_grid(Radial(), [-2, -2, 2, 2], 65)
    rows, polygons = level_polygons(grid, np.exp([1, 3.8, 4.1]))
    assert [r["status"] for r in rows] == ["map_boundary_above_level", "closed", "above_resolved_peak"]
    assert len(polygons) == 1


def test_polygon_extraction_preserves_all_lobes_and_holes():
    x = y = np.linspace(-6, 6, 257)
    xx, yy = np.meshgrid(x, y)
    z = 3-(np.hypot(xx, yy)-3)**2
    rows, polygons = level_polygons((x, y, z), [np.e**2])
    p = polygons[np.e**2]
    assert not p.covers(Point(0, 0))
    assert p.area == pytest.approx(np.pi*(4**2-2**2), rel=.001)
    z = np.maximum(3-((xx-3)**2+yy**2), 3-((xx+3)**2+yy**2))
    rows, polygons = level_polygons((x, y, z), [np.e**2])
    assert rows[0]["components"] == 2
    assert polygons[np.e**2].area == pytest.approx(2*np.pi, rel=.002)


def test_supported_perimeters_cannot_bridge_interior_sampling_gap():
    outer, middle, inner = [Point(0, 0).buffer(r, quad_segs=64) for r in (5, 4, 3)]
    hole = Point(4.4, 0).buffer(.1)
    class Mesh:
        policy = SupportPolicy()
        geometry = outer.difference(hole)
    rows = [{"level_cm": t, "policies": {"standard": {"eligible": True}}} for t in (1., 2., 4.)]
    runs = contiguous_bands(rows, {1.: outer, 2.: middle, 4.: inner}, Mesh())
    assert [r["levels_cm"] for r in runs] == [[1.], [2., 4.]]
    assert not any(r["eligible_for_extension"] for r in runs)


def exponential_areas(t):
    # T=100 exp(-r/10), all fans encoded by summed area, units km and cm.
    return np.pi*100*np.log(100/np.asarray(t))**2


def test_layer_cake_annulus_is_not_vertical_slab_and_brackets_truth():
    t = np.geomspace(2, 20, 1025)
    a = exponential_areas(t)
    r_outer, r_inner = 10*np.log(100/2), 10*np.log(100/20)
    true = 2*np.pi*100*100*((1+r_inner/10)*np.exp(-r_inner/10)
                            -(1+r_outer/10)*np.exp(-r_outer/10))*1e-5
    result = layer_cake_bounds(t, a)
    assert result["lower_km3"] <= true <= result["upper_km3"]
    assert (result["upper_km3"]-result["lower_km3"])/true < .005
    slab = np.trapezoid(a, t)*1e-5
    assert abs(slab/true-1) > .1


def test_matched_exponential_continuation_recovers_known_total():
    t = np.geomspace(2, 20, 1025)
    a = exponential_areas(t)
    b = layer_cake_bounds(t, a)
    band = (b["lower_km3"]+b["upper_km3"])/2
    r = continuation_scenarios(t, a, band)
    matched = next(s for s in r["scenarios"] if s["proximal"] == "exponential_cap" and s["distal"] == "exponential")
    assert matched["total_km3"] == pytest.approx(2*np.pi*100*100*1e-5, rel=1e-5)
    assert r["preferred_global_km3"] is None


def test_proximal_continuation_has_stable_small_cap_limit():
    t = np.geomspace(2, 100*(1-1e-12), 1025)
    result = continuation_scenarios(t, exponential_areas(t), .1)
    constant = next(s["proximal_km3"] for s in result["scenarios"] if s["proximal"] == "constant_cap")
    exponential = next(s["proximal_km3"] for s in result["scenarios"] if s["proximal"] == "exponential_cap")
    assert constant > 0
    assert exponential/constant == pytest.approx(1, abs=1e-10)


def test_band_audit_checks_intermediate_levels_and_preserves_partial_scope():
    from src.empirical_isopach.contour_support import audit_contours
    data = observations(lattice())
    model = Radial()
    report, _, _, _ = audit_contours(model, data, [-8, -8, 8, 8], [model]*4,
        {"standard": SupportMesh(data)}, resolution=129, level_count=9)
    assert len(report["levels"]) == 9 and len(report["refined_levels"]) == 17
    band = report["policies"]["standard"]["selected_band"]
    assert band is not None and band["level_refinement_verified"]
    assert band["volume"]["converged"]
    assert report["preferred_global_km3"] is None


def test_nonintegrable_tail_is_rejected_without_cutoff():
    t = np.geomspace(2, 20, 257)
    # T proportional to x^-1: the outer secant gives p<2.
    a = (100/t)**2
    r = continuation_scenarios(t, a, .1)
    power = [s for s in r["scenarios"] if s["distal"] == "slope_matched_power"]
    assert all(s["total_km3"] is None and s["status"] == "nonintegrable_power_tail" for s in power)


def test_finite_mass_and_endpoint_alone_do_not_bound_distal_volume():
    # Every epsilon>0 is finite, positive and agrees at x=1. Yet volumes diverge.
    eps = np.array([1., .1, .01, .001])
    volume = 2/(2+eps-2)
    assert volume[-1]/volume[0] == pytest.approx(1000)
    with pytest.raises(ValueError):
        layer_cake_bounds([1, 2], [1, 2])


def test_level_selection_is_fixed_and_uses_exact_values_only():
    data = observations(lattice())
    base = candidate_levels(data)
    data.log_value[0] = 10000
    data.censored[0] = True
    assert len(candidate_levels(data)) == 25
    assert np.isfinite(candidate_levels(data)).all()
    assert base[-1] < np.exp(4)
