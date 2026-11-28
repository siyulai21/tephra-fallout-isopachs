"""Integrate the pointwise central 50% band of latent thickness on the map domain."""
import numpy as np
from pathlib import Path
import json
from hashlib import sha256
from scipy.special import logsumexp
from scipy.stats import qmc

from ..core.distributions import predictive_distribution
from ..models.directional.ensemble.fields import inverse_cdf
from .deposit_field import PolarField, polar_nodes
from .deposit_integrals import within_cells

PROBABILITIES = (.25, .5, .75)
DEFINITION = "Integral of each pointwise latent-thickness quantile on the median-map domain"


def field_quantiles(view, xy):
    """Use each map's draw subset, mixture weights and original quantile convention."""
    from ..models.directional.ensemble.maps import MapView as DirectionalView
    from ..models.extent_review import ExtentView
    from ..models.mass_mixture.bearings.maps import BearingView
    from ..models.nested.report import MapView as NestedView

    from ..models.directional.multiscale.posterior import DirectionalField
    if isinstance(view, (DirectionalView,DirectionalField)):
        return view.values(xy, PROBABILITIES)
    values = np.full((3, len(xy)), np.nan)
    inside = np.flatnonzero(view.domain.rho(xy) < view.domain.rho_max)
    for start in range(0, len(inside), 192):
        ix = inside[start:start+192]
        if isinstance(view, ExtentView):
            sample = np.concatenate(view.fields.variants(xy[ix]))
        elif isinstance(view, BearingView):
            sample = np.concatenate(view.field.components(xy[ix]))
        elif isinstance(view, NestedView):
            sample = view.model.field(xy[ix])
        else:
            raise TypeError(f"Unknown posterior map evaluator {type(view).__name__}")
        if isinstance(view, ExtentView) and not view.equal:
            q = inverse_cdf(sample.T, view.weights, PROBABILITIES)
        else:
            q = np.quantile(sample, PROBABILITIES, axis=0)
        values[:, ix] = q / np.log(10.)
    return values


def sample_band(view, data, latitude, shape):
    radii, xy, axis = polar_nodes(data, latitude, *shape)
    q = field_quantiles(view, np.vstack(([0., 0.], xy)))
    if not np.isfinite(q).all() or np.any(np.diff(q,axis=0) < 0):
        raise ValueError("The Earth contains nonfinite or unordered latent thickness quantiles")
    return [PolarField(radii, np.r_[np.repeat(v[0],shape[1]),v[1:]].reshape(shape), axis) for v in q]


def independent_band(view):
    """Check all three surface integrals with two independently scrambled equal-area rules."""
    domain = view.domain
    log_span = np.log1p(domain.rho_max / 50.)
    engines = [qmc.Sobol(2, scramble=True, seed=s) for s in (81173, 31991)]
    chunks = [[], []]
    history = []
    previous_size = 0
    for exponent in (12, 14, 16, 17):
        n = 2**exponent
        integrals, half_integrals = [], []
        for engine, parts in zip(engines, chunks, strict=True):
            u, v = engine.random(n-previous_size).T
            angle = 2*np.pi*u
            radius = 50.*np.expm1(v*log_span)
            xy = np.column_stack((domain.axis_scale*radius*np.sin(angle), radius*np.cos(angle)/domain.axis_scale))
            weights = 1e-5*2*np.pi*log_span*radius*(radius+50.)
            parts.append(weights*10.**field_quantiles(view, xy))
            all_terms = np.concatenate(parts, axis=1)
            integrals.append(all_terms.mean(axis=1))
            half_integrals.append(all_terms[:,:n//2].mean(axis=1))
        centre = np.mean(integrals,axis=0)
        change = np.maximum(np.ptp(integrals,axis=0)/centre, abs(np.mean(half_integrals,axis=0)/centre-1.))
        history.append(dict(points_per_scramble=n, km3=np.asarray(integrals).tolist(),
                            half_grid_km3=np.asarray(half_integrals).tolist(), relative_change=change.tolist()))
        if np.all(change < .01):
            break
        previous_size = n
    return history


def summarize_band(surfaces, independent):
    volumes = np.array([within_cells(s)[0] for s in surfaces])
    coarse = [within_cells(PolarField(s.radii[::2],s.log10_cm[::2,::2],s.axis))[0] for s in surfaces]
    reference = np.mean(independent[-1]["km3"],axis=0)
    disagreement = abs(volumes/reference-1.)
    passed = (np.all(np.isfinite(volumes)) and np.all(np.diff(volumes)>=0)
              and np.all(disagreement < .015) and np.all(abs(coarse/volumes-1.) < .02)
              and max(independent[-1]["relative_change"]) < .01)
    return dict(definition=DEFINITION, probabilities=list(PROBABILITIES), volumes_km3=volumes.tolist(),
                coarse_volumes_km3=coarse, independent_integrals=independent,
                relative_difference=disagreement.tolist(), grid_shape=list(surfaces[0].log10_cm.shape), passed=bool(passed))


def regional_band(model, region, orders=(8, 12, 20, 32, 48, 64, 96), tolerance=.001):
    """Use the same latent distribution and positive-weight hull quadrature as the regional map."""
    history, stable = [], 0
    for order in orders:
        xy, area = region.quadrature(order)
        chunks = []
        for start in range(0,len(xy),1024):
            dist = predictive_distribution(model,xy[start:start+1024])
            chunks.append([dist.quantile(p) for p in PROBABILITIES])
        log_integrals = logsumexp(np.log(area)+np.concatenate(chunks,axis=1),axis=1)+np.log(1e-5)
        if not np.isfinite(log_integrals).all() or max(log_integrals)>700:
            raise ArithmeticError("A regional thickness-quantile integral is not numerically representable")
        volumes = np.exp(log_integrals)
        change = abs(volumes/np.array(history[-1]["km3"])-1) if history else None
        stable = stable+1 if change is not None and max(change) < tolerance else 0
        history.append(dict(order=order, km3=volumes.tolist(), relative_change=change.tolist() if change is not None else None))
        if stable >= 2:
            break
    return dict(definition=DEFINITION, probabilities=list(PROBABILITIES), volumes_km3=volumes.tolist(),
                tolerance=tolerance, refinement=history, passed=stable>=2)



def median_trials(band):
    return [dict(points_per_scramble=t["points_per_scramble"], km3=[v[1] for v in t["km3"]],
                 half_grid_km3=[v[1] for v in t["half_grid_km3"]], relative_change=t["relative_change"][1])
            for t in band["independent_integrals"]]


def checked_band(view, data, latitude):
    from .deposit_field import doubling_levels
    independent = independent_band(view)
    for shape in ((193,240),(385,480)):
        surfaces = sample_band(view,data,latitude,shape)
        field = surfaces[1]
        band = summarize_band(surfaces,independent)
        levels = doubling_levels(10.**field.log10_cm.max())
        _,area = within_cells(field,levels)
        coarse = PolarField(field.radii[::2],field.log10_cm[::2,::2],field.axis)
        _,coarse_area = within_cells(coarse,levels)
        if (not np.any(abs(area-coarse_area)>np.maximum(1.,.05*area)) and band["passed"]
                and band["relative_difference"][1]<=.012):
            break
    rows = np.linspace(0,len(field.radii)-2,11).astype(int)
    cols = np.linspace(0,len(field.angles)-1,11).astype(int)
    x = field.radii[rows]*np.sin(field.angles[cols])*field.axis
    y = field.radii[rows]*np.cos(field.angles[cols])/field.axis
    q = np.array([v.log10_cm[rows,cols] for v in surfaces])
    band["check_points_x_y_km_log10_q25_q50_q75_cm"] = np.column_stack((x,y,q.T)).tolist()
    return field,band


def write_regional_bands(output, models, region, report):
    output = Path(output)
    methods = {k:regional_band(model,region) for k,model in models.items()}
    for k,record in methods.items():
        record["median_map_volume_km3"] = report["methods"][k]["regional_volume"]["median_surface_km3"]
    result = dict(protocol="regional-thickness-band-v1", domain="observation hull",
                  input_sha256=report["input"]["input_sha256"],
                  regional_report_sha256=sha256((output/"report.json").read_bytes()).hexdigest(), methods=methods)
    (output/"thickness-bands.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result
