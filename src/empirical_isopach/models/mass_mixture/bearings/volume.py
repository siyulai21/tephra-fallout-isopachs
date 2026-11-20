"""Positive integration under each fitted spherical mixture, including its distal tails."""
import numpy as np
from scipy.stats import qmc

from ..geometry import RADIUS, centers, from_sphere, kernel_samples
from ..posterior import quantiles
from ..source_geometry import mass_samples
from .field import POWERS, multiplier, phases, selected

PARTS = ("local_footprint","unsampled_proximal","beyond_farthest_site","other_gaps_and_sectors")
SECTORS = ("N","NE","E","SE","S","SW","W","NW")
SCRAMBLES = (39103,75227)


def mass_nodes(draw, u):
    """A unit-mass quadrature under the parent mixture. No rectangular map enters."""
    scale = draw["log_scale"]
    centre = centers(scale,draw["offset"])
    weights = draw["weights"]
    if "core_log_scale" in draw:
        from ..vent_envelope import gaussian_log_z, normalizer
        core = draw["core_log_scale"]
        c = draw.get("fixed_center",centre)
        strength = draw["envelope_strength"]
        z = normalizer(core,scale,c,weights,strength)
        coeff = np.r_[np.exp(gaussian_log_z(RADIUS**2*np.exp(-2*core))-z),
                      strength*weights*np.exp(gaussian_log_z(RADIUS**2*np.exp(-2*scale))-z)]
        group = np.minimum(np.searchsorted(np.cumsum(coeff/coeff.sum()),u[:,0]),len(coeff)-1)
        points,mass = np.empty((len(u),3)),np.full(len(u),coeff.sum())
        for k in np.unique(group):
            m = group==k
            points[m] = kernel_samples(core if k==0 else scale[k-1], [0.,0.,1.] if k==0 else c[k-1],
                                       2.,0.,u[m,1:])
            if k:
                mass[m] *= -np.expm1(-.5*np.sum((points[m]-[0.,0.,1.])**2,axis=1)*RADIUS**2*np.exp(-2*core))
        return points,mass
    group = np.minimum(np.searchsorted(np.cumsum(weights),u[:,0]),len(weights)-1)
    points = np.empty((len(u),3))
    for k in np.unique(group):
        m = group==k
        if "angular" in draw:
            points[m] = mass_samples(scale[k],draw["angular"][k],draw["exponent"],draw["kernel_weights"],u[m,1:],
                                     angular_nodes=max(1024,4*int(draw.get("angular_nodes",256))))
        else:
            points[m] = kernel_samples(scale[k],centre[k],draw["exponent"],draw["tail_fraction"],
                                       u[m,1:],draw.get("exponential_fraction",0.))
    return points,np.ones(len(u))


def integrate_draw(draw,support,phi,u,inner,outer,parts):
    xyz,weight = mass_nodes(draw,u)
    xy = from_sphere(xyz,support.latitude)
    h,theta = support.evaluate(xy)
    # The parent has exactly unit mass. The residual measures quadrature error for the envelope geometry.
    residual = abs(weight.mean()-1)
    weight = weight/weight.mean()
    factors = [weight*multiplier(h,theta,phi,p) for p in POWERS]
    enhanced = weight*multiplier(h,theta,phi,4,enhanced=True)
    levels = np.stack([np.r_[[np.mean(x[:n]) for x in factors],np.mean(enhanced[:n])]
                       for n in (len(u)//2,len(u))])
    groups = None
    if parts:
        radius = RADIUS*np.sqrt(np.maximum(0,2*(1-xyz[:,2])))
        mesh = support.mesh
        tri = mesh.triangulation.find_simplex(xy)
        inside = (tri>=0)&mesh.accepted[np.maximum(tri,0)]
        part = np.where(inside,0,np.where(radius<inner,1,np.where(radius>outer,2,3)))
        sector = np.floor((theta+np.pi/8)/(np.pi/4)).astype(int)%8
        groups = np.stack([np.bincount(8*part+sector,weights=x,minlength=32)/len(x) for x in factors])
    return levels,groups,residual


def volumes(samples,support,*,per_chain=256,power=9,parts=True):
    base,idx,shape = selected(samples,per_chain)
    phi = phases(shape)
    parent = np.exp(base["log_volume"])
    radii = np.linalg.norm(support.mesh.xy*support.scale,axis=1)
    history = []
    for order in range(power,min(power+3,14)):
        us = [qmc.Sobol(4,scramble=True,seed=s).random_base2(order) for s in SCRAMBLES]
        estimates = np.empty((2,2,len(idx),4))
        groups = np.empty((len(idx),3,32)) if parts else None
        largest_residual = 0.
        for i in range(len(idx)):
            draw = {k:v[i] for k,v in base.items()}
            for j,u in enumerate(us):
                estimates[j,:,i],allocation,res = integrate_draw(draw,support,phi[i],u,radii.min(),radii.max(),parts)
                largest_residual = max(largest_residual,res)
                if parts:
                    if j==0:
                        groups[i] = allocation/2
                    else:
                        groups[i] += allocation/2
        fine = estimates[:,1].mean(0)
        full = parent[:,None]*fine[:,:3]
        q = np.quantile(full,[.025,.5,.975])
        q_by_run = np.array([np.quantile(parent[:,None]*e, [.025,.5,.975]) for e in estimates[:,1,:,:3]])
        half = np.quantile(parent[:,None]*estimates[:,0,:,:3].mean(0),[.025,.5,.975])
        delta = np.abs(estimates[0,1]-estimates[1,1])
        test = {"nodes_per_scramble":2**order,"scrambles":SCRAMBLES,
                "max_relative_quantile_scramble_difference":float(np.max(np.abs(q_by_run[0]-q_by_run[1])/q)),
                "max_relative_quantile_doubling_difference":float(np.max(np.abs(half/q-1))),
                "p99_absolute_draw_fraction_difference":float(np.quantile(delta[:,:3],.99)),
                "max_absolute_draw_fraction_difference":float(delta[:,:3].max()),
                "max_parent_mass_error":float(largest_residual)}
        spatial = (max(test["max_relative_quantile_scramble_difference"],test["max_relative_quantile_doubling_difference"])<.025
                   and test["p99_absolute_draw_fraction_difference"]<.06 and largest_residual<.03)
        history.append(test)
        if spatial:
            break
    if not np.allclose(full[:,0],parent) or np.any(full[:,1:]>parent[:,None]*(1+1e-12)):
        raise ArithmeticError("Bearing retention must preserve the identity and positive mass ordering")
    original = np.exp(samples["log_volume"]).ravel()
    base_error = np.max(abs(np.quantile(parent,[.025,.5,.975])/np.quantile(original,[.025,.5,.975])-1))
    half_id = np.tile(np.arange(shape[1])%2==0,shape[0])
    split = np.abs(np.quantile(full[half_id],[.025,.5,.975])-np.quantile(full[~half_id],[.025,.5,.975]))/q
    result = {"volume":{**quantiles(full.ravel()),"unit":"km3 bulk fallout","draws":int(full.size)},
              "by_width":{str(p):quantiles(full[:,j]) for j,p in enumerate(POWERS)},
              "enhanced_hidden_fan":quantiles(parent*fine[:,3]),"base_volume_all_draws":quantiles(original),
              "checks":{"spatial_passed":bool(spatial),"history":history,"parent_subset_quantile_error":float(base_error),
                        "draw_split_relative":split.tolist(),"draw_split_passed":bool(max(split)<.1 and base_error<.1)},
              "chain_count":shape[0],"draws_per_chain":shape[1]}
    packet = {"flat_index":idx,"azimuth_rad":phi,"parent_km3":parent,"volume_km3":full,
              "enhanced_km3":parent*fine[:,3]}
    if parts:
        if not np.allclose(groups.sum(2),fine[:,:3],atol=1e-10):
            raise ArithmeticError("Sector budgets do not conserve mass")
        partition = groups.reshape(len(idx),3,4,8)*parent[:,None,None,None]
        result["mass_partition"] = {"parts":{key:{"volume_km3":quantiles(partition[:,:,j,:].sum(2).ravel()),
             "fraction":quantiles((partition[:,:,j,:].sum(2)/full).ravel())} for j,key in enumerate(PARTS)},
             "sectors":{s:quantiles(partition[:,:,:,i].sum(2).ravel()) for i,s in enumerate(SECTORS)},
             "nearest_exact_site_chord_km":float(radii.min()),"farthest_exact_site_chord_km":float(radii.max())}
        packet["part_sector_km3"] = partition
    return result,packet


def checked_volumes(samples,support,*,parts=True,initial=128,ceiling=1024):
    stages = []
    n = initial
    while True:
        result,packet = volumes(samples,support,per_chain=n,parts=parts)
        stages.append({"draws_per_chain":result["draws_per_chain"],"volume":result["volume"],"checks":result["checks"]})
        if result["checks"]["draw_split_passed"] or n>=min(ceiling,samples["log_volume"].shape[1]):
            break
        n = min(2*n,ceiling,samples["log_volume"].shape[1])
    result["draw_refinements"] = stages
    return result,packet
