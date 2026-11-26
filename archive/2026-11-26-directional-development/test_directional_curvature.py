import numpy as np
import pytest

from src.empirical_isopach.models.directional.multiscale.curvature import (
    covariance, hessians, regularize,
)
from src.empirical_isopach.models.directional.multiscale.geometry import LocalBasis, matern
from src.empirical_isopach.models.mass_mixture.geometry import RADIUS


@pytest.mark.parametrize("kernel, function", [("matern-v5", matern), ("gaussian-mixture", covariance)])
def test_analytic_hessians_against_independent_finite_differences(kernel, function):
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(8, 2))*100
    points = np.vstack((centers[:2], rng.normal(size=(10, 2))*300))
    step = .0001*np.hypot(35, np.linalg.norm(points, axis=1))
    def field(xy):
        return function(centers, xy, .5)*(1-np.sum(xy*xy, axis=1)/(2*RADIUS)**2)
    dx = np.column_stack((step, np.zeros(len(points))))
    dy = dx[:, ::-1]
    numerical = np.array([
        (field(points+dx)+field(points-dx)-2*field(points))/step**2,
        (field(points+dx+dy)+field(points-dx-dy)-field(points+dx-dy)-field(points-dx+dy))/(4*step**2),
        (field(points+dy)+field(points-dy)-2*field(points))/step**2,
    ])
    np.testing.assert_allclose(hessians(centers, points, .5, kernel), numerical,
                               atol=1e-6*np.max(abs(numerical)), rtol=2e-5)


def test_curvature_prior_converges_and_rotates_with_the_sites():
    rng = np.random.default_rng(43)
    xy = rng.normal(size=(10, 2))*90
    query = rng.normal(size=(13, 2))*200
    angle = .432
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    bases = [LocalBasis.fit(points, 1., .5, kernel="gaussian-mixture")
             for points in (xy, xy@rotation)]
    regularized = [regularize(base) for base in bases]
    assert all(b.diagnostics["passed"] for b in regularized)
    features = [b.features(points, 1.) for b, points in zip(regularized, (query, query@rotation))]
    np.testing.assert_allclose(features[0]@features[0].T, features[1]@features[1].T, atol=2e-8)
    np.testing.assert_allclose(regularized[0].unresolved(query, 1.), bases[0].unresolved(query, 1.), atol=1e-12)
    refined = regularize(bases[0], tolerance=.002)
    f = refined.features(query, 1.)
    np.testing.assert_allclose(f@f.T, features[0]@features[0].T, atol=.002)
