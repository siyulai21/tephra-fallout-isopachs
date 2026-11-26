import numpy as np
from src.empirical_isopach.models.directional.multiscale.geometry import LocalBasis

def test_distinct_spatial_amplitudes_survive_posterior_replay(tmp_path):
    from src.empirical_isopach.models.directional.multiscale.geometry import design
    from src.empirical_isopach.models.directional.multiscale.posterior import Component, DirectionalField

    xy = np.array([[2.0, 3.0], [17.0, 10.0], [-35.0, 5.0]])
    local = LocalBasis.fit(xy, 1.0, 0.5)
    parts = []
    for amplitude in (0.2875, 1.15):
        B, _ = design(xy, 1.0, 32.0, 0, local, amplitude)
        parts.append(
            Component(
                0.5, 32.0, 0, 0.5, 4.5, 0.26, 0.2, np.zeros(B.shape[1]), np.eye(B.shape[1]), amplitude
            )
        )
    model = DirectionalField(0.0, {0.5: local}, parts, [])
    path = tmp_path / "field.npz"
    model.save(path)
    restored = DirectionalField.load(path)
    a, b = model.distribution(xy), restored.distribution(xy)
    np.testing.assert_array_equal(a.means, b.means)
    np.testing.assert_array_equal(a.variances, b.variances)
    assert np.all(a.variances[1] > a.variances[0])
    # Observation variance includes the unresolved complement at each amplitude.
    # The sum of coherent and unresolved spatial variance is amplitude squared.
    observed = restored.distribution(xy, observation=True)
    np.testing.assert_allclose(
        observed.variances[1] - observed.variances[0], 1.15**2 - 0.2875**2, rtol=3e-5
    )

