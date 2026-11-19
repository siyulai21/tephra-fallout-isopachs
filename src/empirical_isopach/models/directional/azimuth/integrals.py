"""Whole-Earth quadrature for paired median maps and complete, coherent deposits."""
import numpy as np
from scipy.stats import qmc

from ..ensemble.fields import inverse_cdf
from ..ensemble.integrals import interval, polar_nodes, SEEDS, ORDERS
from ..geometry import polar

LEVELS = (.1, 1., 10., 30., 100.)
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
PARTS = ("measured triangles", "transition collar", "proximal", "between observed radii", "distal")


def split_weights(model):
    even = np.concatenate([np.tile(np.arange(len(m.index))%2 == 0, m.draws["source"].shape[0])
                           for m in model.members])
    return model.weights*even, model.weights*~even


def partition(g, xy, radius, theta, inner, outer):
    triangles, _, _ = g.locate(xy)
    part = np.where(radius < inner, 2, np.where(radius <= outer, 3, 4))
    hit = triangles >= 0
    part[hit] = np.where(g.inside[triangles[hit]], 0, 1)
    sectors = np.floor((theta+np.pi/8)/(np.pi/4)).astype(int)%8
    return 8*part+sectors, sectors


def integrate(model, conditioned=None):
    g = model.members[0].geometry
    limits = np.quantile(polar(g.mesh.xy, g.axis)[0], [0.,1.])
    weights = [model.weights, *split_weights(model)]
    if conditioned is not None:
        weights.append(conditioned)
    generators = [qmc.Sobol(d=2,scramble=True,seed=s) for s in SEEDS]
    maps = np.zeros((2, 1+len(weights), 40))
    draws = np.zeros((2,len(model.weights)))
    areas = np.zeros((2,2,len(LEVELS),8))
    compare = np.zeros((2,len(LEVELS),3))
    history, previous = [], None
    for i,power in enumerate(ORDERS):
        n = 2**power
        for rep,gen in enumerate(generators):
            points = gen.random_base2(power if i==0 else power-1)
            for start in range(0,len(points),128):
                xy,area,r,theta = polar_nodes(points[start:start+128],g.axis)
                group,sector = partition(g,xy,r,theta,*limits)
                original,revised,_ = model.components(xy)
                revised = np.ascontiguousarray(revised.T)
                order = np.argsort(revised,axis=1)
                fields = [np.exp(inverse_cdf(original.T,model.base.weights)[0])]
                fields.extend(np.exp(inverse_cdf(revised,w,order=order)[0]) for w in weights)
                for target,z in zip(maps[rep],fields,strict=True):
                    target += np.bincount(group,weights=z*area*1e-5,minlength=40)
                draws[rep] += (area*1e-5) @ np.exp(revised)
                for j,level in enumerate(LEVELS):
                    for jmodel in range(2):
                        areas[rep,jmodel,j] += np.bincount(sector,weights=area*(fields[jmodel]>=level),minlength=8)
                    union = np.maximum.reduce(fields[1:4]) >= level
                    compare[rep,j] += [area@((fields[2]>=level)!=(fields[3]>=level)),
                                       area@union,(area**2)@union]
        avg = maps.mean(0)/n
        central = avg[1].sum()
        spread = np.abs(maps[0,1]-maps[1,1]).sum()/(n*central)
        change = float(np.abs(avg[1]-previous).sum()/central) if previous is not None else None
        mass = draws.mean(0)/n
        quant = interval(mass,model.weights)
        qq = np.array([list(interval(a/n,model.weights).values()) for a in draws])
        posterior_change = float(np.max(np.abs(qq[0]-qq[1])/np.maximum(qq.mean(0),1e-20)))
        split = float(max(abs(avg[j].sum()/central-1) for j in (2,3)))
        discrepancies = []
        for level,(difference,union,sq) in zip(LEVELS,compare.sum(0),strict=True):
            eff = union**2/sq if sq else 0.
            discrepancies.append({"level_cm":level,"effective_union_nodes":float(eff),
                                  "symmetric_difference_fraction":float(difference/union) if union else None})
        history.append({"points_per_scramble":n,"central_km3":float(central),"partition_scramble_L1":float(spread),
                        "partition_nested_L1":change,"joint_quantile_scramble_change":posterior_change,
                        "split_median_volume_change":split})
        spatial = change is not None and change<.015 and spread<.025 and posterior_change<.025
        if spatial:
            break
        previous = avg[1].copy()
    split_ok = split<.025 and all(a["symmetric_difference_fraction"]<.1 for a in discrepancies
                                  if a["effective_union_nodes"]>=64)
    split_joint = np.array([list(interval(mass,w).values()) for w in weights[1:3]])
    joint_split = np.max(abs(split_joint/np.array(list(quant.values()))-1),axis=0)
    report = {"median_map_km3":float(central),"matched_control_map_km3":float(avg[0].sum()),
              "coherent_volume_km3":quant,"joint_quantile_split_relative":joint_split.tolist(),
              "parts":PARTS,"sectors":SECTORS,"median_map_parts_by_sector_km3":avg[1].reshape(5,8).tolist(),
              "control_parts_by_sector_km3":avg[0].reshape(5,8).tolist(),"level_cm":LEVELS,
              "area_by_sector_km2":dict(zip(("control","revised"),(areas.mean(0)/n).tolist(),strict=True)),
              "numerical":{"scramble_seeds":SEEDS,"history":history,"spatial_passed":bool(spatial),
                           "split_map_passed":bool(split_ok),"split_areas":discrepancies,
                           "joint_quantile_split_passed":bool(max(joint_split)<.1)}}
    if conditioned is not None:
        report["archive_conditioned"] = {"median_map_km3":float(avg[-1].sum()),
                                         "coherent_volume_km3":interval(mass,conditioned)}
    return report,mass


def directional_reach(model):
    """First down-ray crossing, never jump across a gap to a detached lobe."""
    from ..geometry import RADIUS
    axis = model.members[0].geometry.axis
    r = np.r_[0., np.geomspace(.1,2*RADIUS,400)]
    result = []
    for i,name in enumerate(SECTORS):
        angle = i*np.pi/4
        xy = r[:,None]*[axis*np.sin(angle),np.cos(angle)/axis]
        original,new,_ = model.components(xy)
        for label,field,w in (("control",original,model.base.weights),("revised",new,model.weights)):
            median = inverse_cdf(field.T,w)[0]
            for cm in LEVELS:
                level = np.log(cm)
                below = np.flatnonzero(median<level)
                if not len(below):
                    crossing = r[-1]
                elif below[0]==0:
                    crossing = 0.
                else:
                    j = below[0]
                    crossing = np.interp(level,median[[j,j-1]],r[[j,j-1]])
                result.append({"method":label,"sector":name,"level_cm":cm,
                               "first_crossing_great_circle_km":float(2*RADIUS*np.arcsin(min(1.,crossing/(2*RADIUS))))})
    return result
