"""Checks on geographic integration, posterior resolution and structural sensitivity."""
import numpy as np

from ....contour_support import audit_contours
from ....core.geometry import survey_bounds
from ..continuation import join
from ..geometry import positions, ANGLES
from ..reach_integrals import nodes
from ..report import MapView
from . import geometry
from src.probabilistic_isopach.global_volume import EarthDomain


def population_field_check(model,data):
    """Compare independent posterior medians between and beyond the field sites.

    This is sampling repeatability of the same posterior, not a claim that an
    unobserved direction is geologically identified. Fixed annuli and bearings
    include the proximal transition and the complete physical chart.
    """
    radius,_=positions(data.xy,model.latitude)
    positive=radius[radius>0]
    first=max(.01,float(positive.min())/4.) if len(positive) else .01
    domain=EarthDomain.wgs84(model.latitude)
    r=np.geomspace(first,domain.rho_max*(1.-1e-8),32)
    theta=np.arange(48)*2*np.pi/48
    xy=np.column_stack((np.outer(r,np.sin(theta)).ravel()*domain.axis_scale,
                        np.outer(r,np.cos(theta)).ravel()/domain.axis_scale))
    median=[]
    for indices in np.array_split(np.arange(len(model.z)),4):
        part=model.subset(indices)
        median.append(np.concatenate([np.median(part.field(xy[i:i+64]),axis=0)
                                      for i in range(0,len(xy),64)]))
    median=np.array(median)
    visible=median.max(axis=0)>=np.log(.5)
    spread=np.ptp(median[:,visible],axis=0)
    middle,high=np.quantile(spread,[.5,.95]) if len(spread) else (0.,0.)
    return dict(points=len(xy),visible_points=int(visible.sum()),minimum_cm=.5,
                median_log_cm=float(middle),p95_log_cm=float(high),
                passed=bool(middle<.15 and high<.4))


def integral_check(model):
    sample = model.reduced(192)
    reference = sample.volume()
    history = []
    for order in (13, 15, 17):
        errors = []
        for seed in (70139, 31907):
            xy, weight, _, _ = nodes(order, seed, model.latitude)
            total = np.zeros(len(reference))
            for start in range(0, len(xy), 256):
                sl = slice(start,start+256)
                total += np.exp(sample.field(xy[sl])) @ weight[sl]
            errors.extend(abs(1.e-5*total/(len(xy)*reference)-1.))
        median, p95, maximum = np.quantile(errors,[.5,.95,1.])
        passed = bool(median < .01 and p95 < .05)
        history.append(dict(points_per_scramble=2**order, scrambles=[70139,31907],draws=len(sample.z),
                            median_error=median,p95_error=p95,maximum_error=maximum,passed=passed))
        if passed:
            break
    return history


def map_draws(model):
    grids = [nodes(12,s,model.latitude)[:2] for s in (18971,42533)]
    history = []
    old = None
    for n in (256,512,1024,2048,4096):
        n = min(n,len(model.z))
        view = MapView(model.reduced(n))
        fields = [view.central(xy) for xy,_ in grids]
        volume = float(np.mean([1.e-5*np.mean(w*10.**f) for (_,w),f in zip(grids,fields,strict=True)]))
        change, shape = None, None
        if old is not None:
            change = abs(volume/history[-1]["median_map_km3"]-1.)
            # Resolve angular structure at map-visible thickness, including the outer 0.5 cm contour.
            both = np.stack((np.concatenate(old), np.concatenate(fields)))
            visible = both.max(axis=0) >= np.log10(.25)
            shape = float(np.quantile(abs(np.diff(both[:,visible],axis=0)),.95)) if visible.any() else 0.
        passed = bool(change is not None and change < .025 and shape < .08)
        history.append(dict(draws=n,median_map_km3=volume,relative_change=change,
                            p95_abs_log10_change_visible=shape,passed=passed))
        if passed or n == len(model.z):
            break
        old = fields
    return history


def weight_check(volumes, ratios):
    order = np.argsort(volumes)
    weight = ratios / np.sum(ratios)
    q = np.interp([.025,.5,.975],np.cumsum(weight[order]),volumes[order])
    effective = 1./np.sum(weight**2)
    return dict(complete_draw_quantiles_km3=q.tolist(),relative_ess=float(effective/len(weight)),
                effective_draws=float(effective),max_weight=float(weight.max()))


def sensitivity(model, data):
    geometry = model.geography
    indices = model.geometry//ANGLES
    family = np.array([s.family for s in geometry.SHAPES])[indices]
    core = np.array([s.core_km for s in geometry.SHAPES])[indices]
    v = model.volume()
    cases = {"equal_family_odds":np.array([.2/geometry.FAMILY_PRIOR[s] for s in family]),
             "fourfold_complex_phase_odds":np.where(np.isin(family,["two phases","turning","detached"]),4.,1.),
             "quarter_complex_phase_odds":np.where(np.isin(family,["two phases","turning","detached"]),.25,1.)}
    # Tilt the core scales within their original family, leaving each family's mass fixed.
    varied = np.isin(family,["single","turning","two phases"])
    for sign,label in ((-1,"shorter_migration_scale"),(1,"longer_migration_scale")):
        factors = (np.array(geometry.CORES)/32.)**sign
        cases[label] = np.where(varied,(core/32.)**sign/factors.mean(),1.)
    secondary = np.array([s.distal_ratio for s in geometry.SHAPES])[indices]
    two = family == "two phases"
    for factor in (.25,4.):
        normalizer = (2.+factor)/3.
        cases[f"persistent_second_phase_prior_x{factor:g}"] = np.where(two,np.where(secondary==.8,factor,1.)/normalizer,1.)
    result = {key:weight_check(v,w) for key,w in cases.items()}
    if getattr(model,"geography_version",2) == 4:
        # The second normal latent controls physical bending. Orientation jitter
        # completes the uniform bearing prior and is not a curvature parameter.
        for scale in (.5,2.):
            ratio=np.exp(.5*model.warp[:,1]**2*(1.-1./scale**2))/scale
            result[f"bend_latent_prior_sd_x{scale:g}"]=weight_check(v,ratio)
    result["second_phase"] = {"probability":float(two.mean()),
        "conditional_distal_ratio_probabilities":{str(f):float(np.mean(secondary[two]==f)) for f in (0.,.3,.8)} if two.any() else {},
        "conditional_volume_quantiles_km3":{str(f):np.quantile(v[two & (secondary==f)],[.025,.5,.975]).tolist()
                                          for f in (0.,.3,.8) if np.any(two & (secondary==f))}}
    for factor in (.25,4.):
        moved = (model.recontinue(data,tilt=factor) if hasattr(model,"recontinue") else
                 join(model,data,donor=model.donor,transfer=model.transfer,tilt=factor,ranker=geometry.ranks))
        result[f"outer_transfer_x{factor:g}"] = {"complete_draw_quantiles_km3":np.quantile(moved.volume(),[.025,.5,.975]).tolist()}
    if hasattr(model,"ranks_at"):
        rank = model.ranks_at(data.xy)
        inv = np.arange(len(model.z))
    else:
        unique, inv = np.unique(model.geometry,return_inverse=True)
        rank = geometry.ranks(*positions(data.xy,model.latitude),unique)
    exact = ~data.censored
    positive = exact | data.lower_censored
    lo = rank[:,exact].min(axis=1)[inv]
    far = rank[:,positive].max(axis=1)[inv]
    partition = np.stack((model.volume_at(lo), model.volume_at(far)-model.volume_at(lo), v-model.volume_at(far)))
    result["rank_partitions"] = {"groups":["inside_first_exact_rank","between_first_exact_and_last_positive_rank","outside_last_positive_rank"],
                                 "median_mass_fractions":np.median(partition/v,axis=1).tolist(),
                                 "outside_last_positive_fraction95":np.quantile(partition[-1]/v,[.025,.975]).tolist(),
                                 "interpretation":"These are fitted whole-area ranks. A measured site does not verify the rest of that perimeter."}
    return result


def support(data, meta, full, fold_models):
    bounds = survey_bounds(data.xy)
    domain = EarthDomain.wgs84(meta["vent_lonlat"][1])
    corners = np.array([[x,y] for x in bounds[[0,2]] for y in bounds[[1,3]]])
    if domain.rho(corners).max() >= domain.rho_max:
        return dict(status="audit_window_leaves_physical_chart", policies={},refined_levels=[])
    valid = [m.reduced(256) for m,ok in fold_models if ok]
    audit,*_ = audit_contours(full.reduced(256),data,bounds,valid)
    return {**audit,"status":"completed","deletion_checks":[{"fold":n,"passed":bool(ok)} for n,(_,ok) in enumerate(fold_models)],
            "map_bounds_km":bounds.tolist(),"draws_per_fold_cap":256}
