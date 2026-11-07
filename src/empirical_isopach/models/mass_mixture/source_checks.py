"""Report actual source constraints and numerical mass-normalization accuracy."""
import numpy as np
from .inference import flatten
from .geometry import log_field
from .source_geometry import normalizers


def source_checks(samples):
    flat = flatten(samples)
    vent = np.exp(log_field(flat, np.array([[0., 0., 1.]]))[:, 0])
    result = {'source_anchored': 'angular' in flat,
              'vent_thickness_cm': dict(zip(('lower95', 'median', 'upper95'),
                                            np.quantile(vent, [.025, .5, .975]).tolist()))}
    if 'core_log_scale' in flat:
        return {**result, 'source_anchored': True, 'maximum_at_vent_guaranteed': True,
            'nonincreasing_along_every_source_ray': False,
            'secondary_maxima_allowed': True,
            'centers_at_measured_locations': 'fixed_center' in flat,
            'interpretation': 'The source is the unique global maximum by the bounded-envelope identity. Smaller secondary peaks remain possible. This is a shape assumption, not a vent pseudo-observation.',
            'normalization': {'method': 'Analytical whole-Earth integrals of Gaussian profiles and their products',
                              'passed': True, 'quadrature_used_for_total': False}}
    if 'angular' not in flat:
        return {**result, 'maximum_at_vent_guaranteed': False,
                'normalization': 'Analytical circular kernels; component centres can move away from the vent.'}
    errors, last_errors = [], []
    nodes = int(np.unique(flat.get('angular_nodes', [256]))[0])
    for start in range(0, len(vent), 128):
        ls, angular, q = (flat['log_scale'][start:start+128], flat['angular'][start:start+128],
                          flat['exponent'][start:start+128, None])
        a = normalizers(ls, angular, q, nodes)
        b = normalizers(ls, angular, q, 2*nodes)
        c = normalizers(ls, angular, q, 4*nodes)
        errors.extend(np.max(abs(np.expm1(c-a)), axis=(1, 2)).tolist())
        last_errors.extend(np.max(abs(np.expm1(c-b)), axis=(1, 2)).tolist())
    maximum = float(np.max(errors))
    return {**result, 'maximum_at_vent_guaranteed': True,
        'nonincreasing_along_every_source_ray': True,
        'interpretation': 'These are explicit shape constraints, not proof that the true deposit has no secondary maxima. No vent thickness observation or pseudo-data were inserted.',
        'normalization': {'radial_integral': 'analytical over the full Earth',
            'angular_nodes_fitted': nodes, 'angular_nodes_checked': 4*nodes,
            'posterior_fields_checked': len(vent),
            'max_relative_component_mass_error': maximum,
            'max_relative_last_refinement_change': float(np.max(last_errors)),
            'relative_tolerance': 1e-6, 'passed': maximum < 1e-6}}
