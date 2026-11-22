import numpy as np

from src.empirical_isopach.core.observations import Observations
from src.empirical_isopach.models.nested.continuation import join
from src.empirical_isopach.models.nested.geometry import positions
from src.empirical_isopach.models.nested.inference import Deposit
from src.empirical_isopach.models.nested.reach import Corridors, PRIOR, ReachDraws, condition
from src.empirical_isopach.models.nested.reach_integrals import integrate


def survey():
    points = np.array([[8., 0.], [18., 2.], [75., 4.], [450., 1.], [140., 32.], [27., -12.]])
    return Observations(points, np.log([55.,41.,22.,1.,4.,31.]), np.zeros(6,bool), np.zeros(6), np.arange(6))


def reference(data):
    n = 4096
    z = np.zeros((n,6))
    base = Deposit(z, np.zeros(n,int), np.zeros(n,bool), 0.)
    return join(base, data)


def test_widths_in_each_independent_population_and_serialization(tmp_path):
    data = survey()
    previous = reference(data)
    model, check = condition(previous, data)
    np.testing.assert_allclose(previous.field(data.xy), model.field(data.xy), atol=1e-10)
    for block in np.array_split(model.law, 4):
        frac = np.bincount(block, minlength=len(PRIOR))/len(block)
        np.testing.assert_allclose(frac, PRIOR, atol=.047)
    assert check["passed"]
    file = tmp_path / "packet.npz"
    model.save(file)
    other = ReachDraws.load(file)
    np.testing.assert_array_equal(other.law, model.law)
    np.testing.assert_array_equal(other.field(data.xy[:2]), model.field(data.xy[:2]))


def test_crosswind_silence_is_conditional_and_rotationally_equivariant():
    data = survey()
    corridors = Corridors.from_training(data, 0.)
    r, t = positions(data.xy[[3]], 0.)
    np.testing.assert_allclose(corridors.penalties(r, t), 0., atol=1e-12)
    far_side = corridors.penalties(r, t+np.pi)
    assert far_side[0,0] == 0. and np.all(far_side[1:] < -2)
    assert np.isfinite(far_side).all()
    rotated = Corridors(corridors.radii, corridors.bearings+.79, corridors.proximal)
    np.testing.assert_allclose(rotated.penalties(r,t+.79+np.pi), far_side, atol=1e-12)
    near = corridors.penalties(r*.8,t)
    assert np.all(near > -1e-10)
    beyond = corridors.penalties(r*2, t)
    assert np.all(beyond[1:] < 0.)


def test_upper_bound_updates_likelihood_without_adding_corridor():
    data = survey()
    xy = np.vstack((data.xy, [-125., 0.]))
    bound = Observations(xy, np.r_[data.log_value,np.log(.06)], np.r_[data.censored,True],
                         np.zeros(len(xy)), np.arange(len(xy)))
    previous = reference(data)
    model, audit = condition(previous,bound,positive_proposal=True)
    assert audit["upper_rows"] == [6] and audit["positive_rows"] == list(range(6))
    assert model.corridors.radii.shape == (6,)
    assert np.median(model.field(xy[-1:])) < np.median(previous.field(xy[-1:]))
    assert audit["populations"][0]["law_probabilities"][0] < PRIOR[0]


def test_all_bearing_integral_matches_analytical_parent():
    data = survey()
    parent = reference(data).reduced(12)
    model = ReachDraws(parent, np.zeros(len(parent.z), int), Corridors.from_training(data, 0.))
    mass, _, _, _, sampled_reference, _ = integrate(model,data,13,28439)
    np.testing.assert_allclose(mass.sum(axis=1), sampled_reference, rtol=1e-12)
    np.testing.assert_allclose(sampled_reference, parent.volume(), rtol=.02)
