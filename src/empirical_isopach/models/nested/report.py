"""Whole-deposit uncertainty, support and independently integrated geographic fields."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t
from scipy.stats import qmc

from ...contour_support import audit_contours
from ...core.audit import audited_data
from ...core.catalog import dataset_record
from ...core.geometry import survey_bounds
from ...support import SupportMesh
from .continuation import PROTOCOL, Continued, continuation_digest, join, from_parent, load_packet as load_draws, radius_for, slab_for, volume_for
from .distal import evaluate as distal_sensitivity
from .diagnostics import allocation_info, check_bounds, classes, edge_sites, predictions
from ...visualization.global_maps import world_map
from ...visualization.maps import (frame_bounds,grid_axes,observation_legend,observations,setup_map)
from ..directional.archive_readings import load_inventory,positions as archive_positions
from ..directional.ensemble.fields import inverse_cdf
from ..directional.ensemble.integrals import polar_nodes
from .geometry import DIAMETER, SHAPES, ANGLES, geometry_probabilities, positions
from .inference import probabilities
from .profile import ASSET, CORE, RATIO, DONOR_ODDS, ODDS_CENTRE, ODDS_SPREAD, above_slab, contour_radius, parameters
from .study import digest, engine_digest, infer_dataset, write_json
from src.probabilistic_isopach.global_volume import EarthDomain

PROBS = (.025,.5,.975)
LEVELS = (.01,.1,1.,10.,100.,1000.)


def area_check(model, data):
    part = model.reduced(256)
    expected = part.volume()
    axis = EarthDomain.wgs84(part.latitude).axis_scale
    mesh = SupportMesh(data)
    exact_radius,_ = positions(data.xy[~data.censored],part.latitude)
    limits = [exact_radius.min(),exact_radius.max()]
    histories = []
    for order in (13,15,17):
        ratios = []
        partitions = []
        for seed in (53917,83129):
            uv = qmc.Sobol(2,scramble=True,seed=seed).random_base2(order)
            groups = np.zeros((len(expected),32))
            for batch in np.array_split(uv,len(uv)//256):
                xy,weight,r,theta = polar_nodes(batch,axis)
                group = classes(mesh,xy,r,theta,limits)
                mass = 1e-5*np.exp(part.field(xy))*weight
                for k in np.unique(group):
                    groups[:,k] += mass[:,group==k].sum(axis=1)
            partitions.append(groups/len(uv))
            ratios.append(groups.sum(axis=1)/len(uv)/expected)
        error = np.abs(np.array(ratios)-1.)
        passed = bool(np.quantile(error,.95)<.05 and np.median(error)<.01)
        histories.append({"nodes_per_scramble":len(uv),"median_relative_error":float(np.median(error)),
                          "q95_relative_error":float(np.quantile(error,.95)),"maximum_relative_error":float(error.max()),
                          "passed":passed})
        if passed:
            break
    return histories, allocation_info(np.mean(partitions,axis=0),expected)


def sensitivities(model):
    z,tail = model.z,model.power_tail
    # Exact prior density ratios on the continuous latent coordinates. All
    # target observations and their likelihood remain the same in these scenarios.
    delta_core,delta_ratio = -CORE/.9,-RATIO
    no_transfer = delta_core*z[:,2]-.5*delta_core**2 + (~tail)*(delta_ratio*z[:,3]-.5*delta_ratio**2)
    wide_cap = -.5*(z[:,0]*2.8/4.)**2+.5*z[:,0]**2+np.log(2.8/4.)
    families = np.array([s.family for s in SHAPES])[model.geometry//ANGLES]
    geometry_prior = geometry_probabilities().reshape(len(SHAPES),ANGLES).sum(axis=1)
    detached_prior = np.sum([w for w,s in zip(geometry_prior,SHAPES,strict=True) if s.family=="detached"])
    scenarios = {"baseline":np.zeros(len(z)),"weak_profile_transfer":no_transfer,
                 "broader_peak_prior":wide_cap,"power_tail_prior_80pct":np.log(np.where(tail,1.6,.4)),
                 "exponential_tail_prior_80pct":np.log(np.where(tail,.4,1.6)),
                 "detached_prior_50pct":np.log(np.where(families=="detached",.5/detached_prior,.5/(1-detached_prior)))}
    if z.shape[1]==6:
        log_odds = ODDS_CENTRE + ODDS_SPREAD*z[:,3]
        base_log = student_t.logpdf(z[:,3],3)-np.log(ODDS_SPREAD)
        def odds_change(centre,spread,df=3):
            return student_t.logpdf((log_odds-centre)/spread,df)-np.log(spread)-base_log
        scenarios["weak_profile_transfer"] = delta_core*z[:,2]-.5*delta_core**2+odds_change(0.,2*ODDS_SPREAD)
        for direction,shift in [("more",np.log(4.)),("less",-np.log(4.))]:
            scenarios[direction+"_distal_ash_prior"] = odds_change(ODDS_CENTRE+shift,ODDS_SPREAD)
        for held in DONOR_ODDS:
            donor = np.log([v for k,v in DONOR_ODDS.items() if k!=held])
            scenarios["distal_prior_without_"+held] = odds_change(donor.mean(),donor.std(ddof=1)*np.sqrt(1+1/len(donor)),len(donor)-1)
    result = {}
    for name,lw in scenarios.items():
        w = probabilities(lw)
        ess = 1/(w@w)
        q = np.quantile(model.volume(),PROBS) if name=="baseline" else inverse_cdf(model.volume(),w,PROBS)
        result[name] = {"volume_km3":q.tolist(),
                        "effective_draws":float(ess),"relative_ess":float(ess/len(w)),
                        "weight_check_passed":bool(ess>=100 and ess/len(w)>=.05)}
    return result


def archive_predictions(path, archives, data, meta, model):
    inventory,file = load_inventory(path,archives)
    if not inventory["sites"]:
        return {}
    sites = inventory["sites"]
    xy = archive_positions(sites,meta["vent_lonlat"])
    mu = model.field(xy)
    noise = parameters(model.z)[-1]
    ids = [s["id"] for s in sites]
    readings = {sid:{"reading":s["reading"],"latent_cm":np.exp(np.quantile(mu[:,i],PROBS)).tolist(),
                     "coincident_with_training":bool(np.linalg.norm(data.xy-xy[i],axis=1).min()<1e-6)}
                for i,(sid,s) in enumerate(zip(ids,sites,strict=True))}
    scenarios = {}
    for scenario in inventory["scenarios"]:
        probabilities_at_sites = {}
        for sid,threshold in scenario["effective_limit_cm"].items():
            v = (np.log(threshold)-mu[:,ids.index(sid)])/noise
            probabilities_at_sites[sid] = float(np.mean(.95*ndtr(v)+.05*ndtr(v/3.)))
        scenarios[scenario["id"]] = {"assumption":scenario["assumption"],
            "effective_scale_cm":scenario["effective_limit_cm"],"probability_below":probabilities_at_sites}
    return {"inventory_sha256":digest(file),"sites":readings,"scenarios":scenarios,
            "usage":"Predictions only. Coincident bounds already contribute once to the accepted quantitative likelihood."}


class MapView:
    statistic = "Pointwise median of nested deposits"

    def __init__(self, model):
        self.model = model
        self.domain = EarthDomain.wgs84(model.latitude)

    def central(self,xy):
        out = np.full(len(xy),np.nan)
        ids = np.flatnonzero(np.isfinite(xy).all(axis=1) & (self.domain.rho(xy)<self.domain.rho_max))
        for idx in np.array_split(ids,max(1,(len(ids)+255)//256)):
            if len(idx):
                out[idx] = np.median(self.model.field(xy[idx]),axis=0)/np.log(10.)
        return out


def figures(folder,name,data,meta,model):
    view = MapView(model.reduced(512))
    frame = frame_bounds(data.xy)
    x,y = grid_axes(frame,longest=211)
    xx,yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(),yy.ravel()))
    z = view.central(xy).reshape(xx.shape)
    norm = Normalize(-1.,max(1.,np.ceil(np.nanmax(z))))
    fig,(left,right) = plt.subplots(1,2,figsize=(12,5.6),layout="constrained")
    image = left.pcolormesh(x,y,z,norm=norm,cmap="viridis",shading="auto",rasterized=True)
    levels = [v for v in (-1,0,1,2,3) if np.nanmin(z)<v<np.nanmax(z)]
    if levels:
        contours = left.contour(x,y,z,levels=levels,colors="white",linestyles="dashed",linewidths=1.)
        left.clabel(contours,fmt=lambda v:f"{10**v:g}",fontsize=8)
    setup_map(left,frame,meta["vent_lonlat"])
    observations(left,data,meta=meta,size=16)
    left.set_title("Median isopachs in the survey area")
    fig.colorbar(image,ax=left,shrink=.74,label="log₁₀ thickness (cm)")
    low,high = np.exp(np.quantile(data.log_value,[0,1]))
    ts = np.geomspace(min(.01,low*.1),max(1000.,high*3),100)
    radii = np.stack([radius_for(model,t) for t in ts])
    areas = np.pi*radii**2
    band = np.quantile(areas,PROBS,axis=1)
    positive = band[1]>0
    right.fill_between(ts[positive],np.where(band[0,positive]>0,band[0,positive],np.nan),
                       band[2,positive],color="#9bc9cb",alpha=.65)
    right.plot(ts[positive],band[1,positive],color="#086f78",lw=2)
    right.set(xscale="log",yscale="log",xlabel="Thickness threshold (cm)",ylabel="Total area at or above threshold (km²)",
              title="Whole-Earth isopach area")
    right.grid(alpha=.15,which="both")
    fig.suptitle(name+" · Bayesian nested isopachs")
    observation_legend(fig,data,meta=meta)
    fig.savefig(folder/"isopachs.png",dpi=145)
    plt.close(fig)
    return world_map(folder,name,data,meta,view,norm)


def whole_area_dataset(path, output, name=None, *, archives=None, resume=False, budget=True, base_cache=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Use a fresh nested-isopach output or verified resume")
    data,meta = audited_data(path,allow_intervals=True)
    if resume:
        designs = [(output,PROTOCOL if budget else "nested-isopach-v1",continuation_digest() if budget else engine_digest())]
        if budget:
            designs.append((output/"profile-fit","nested-budget-v2",engine_digest()))
        for folder,protocol,code_hash in designs:
            for name_part in ("full","fold-0","fold-1","fold-2","fold-3"):
                receipt = folder/(name_part+".json")
                if receipt.is_file():
                    row = json.loads(receipt.read_text())
                    identity = row["identity"]
                    if (identity["input_sha256"],identity["protocol"],identity["engine_sha256"]) != (meta["input_sha256"],protocol,code_hash):
                        raise ValueError(f"Resume inputs or model differ from {receipt}. Choose a fresh output directory.")
                    packet = receipt.with_suffix(".npz")
                    if not packet.is_file() or digest(packet) != row["posterior_sha256"]:
                        raise ValueError(f"Saved posterior differs from {receipt}. Choose a fresh output directory.")
    if budget:
        base = Path(base_cache) if base_cache is not None else output/"profile-fit"
        if base_cache is None:
            infer_dataset(path,base,budget=True)
        folds = from_parent(data,meta,base,output)
    else:
        folds = infer_dataset(path,output,budget=False)
    name = name or (dataset_record(path) or {}).get("name",Path(path).stem)
    full = load_draws(output/"full.npz")
    check = json.loads((output/"full.json").read_text())
    bounds = survey_bounds(data.xy)
    corners = np.array([[a,b] for a in bounds[[0,2]] for b in bounds[[1,3]]])
    if EarthDomain.wgs84(full.latitude).rho(corners).max()<DIAMETER:
        deletions = [load_draws(output/f"fold-{f['fold']}.npz").reduced(256)
                     for f in folds["folds"] if f["numerics"]["passed"]]
        screen,*_ = audit_contours(full.reduced(256),data,bounds,deletions)
        screen["status"] = "completed"
    else:
        screen = {"status":"audit_window_leaves_physical_chart","policies":{}}
    screen["numerically_usable_geographic_folds"] = [f["fold"] for f in folds["folds"] if f["numerics"]["passed"]]
    write_json(output/"support.json",screen)
    volume = full.volume()
    above = np.array([volume,*[slab_for(full,t) for t in LEVELS],
                      np.zeros(len(volume))])
    slabs = -np.diff(above,axis=0)
    if np.min(slabs)<-1e-7*max(1,volume.max()) or not np.allclose(slabs.sum(axis=0),volume):
        raise ArithmeticError("Thickness slabs must be nonnegative and conserve each draw's volume")
    np.savez_compressed(output/"volume-budget.npz",volume_km3=volume,level_cm=LEVELS,slab_km3=slabs)
    area = {str(t):np.quantile(np.pi*radius_for(full,t)**2,PROBS).tolist() for t in LEVELS}
    contours = [k for k,v in screen["policies"].items() if v["selected_band"]]
    support_note = ("A complete contour band passed under " + ", ".join(contours) + "." if contours
                    else "No complete contour band passed the shared support screen.")
    direct,allocation = area_check(full,data)
    bounds_check = check_bounds(data,meta,full)
    coverage = predictions(output,data,folds)
    distal = distal_sensitivity(output,data,folds)
    report = {"protocol":check["identity"]["protocol"],"name":name,"input":{k:v for k,v in meta.items() if k!="transform_inv"},
              "thinning_source_sha256":digest(ASSET),"posterior_sha256":digest(output/"full.npz"),
              "draws":len(volume),"conditional_volume_km3":np.quantile(volume,PROBS).tolist(),
              "population_check_passed":check["numerics"]["passed"],"area_check":direct,"spatial_volume_budget":allocation,"exact_site_coverage95":coverage,
              "prior_sensitivity":sensitivities(full),"area_km2_at_cm":area,
              "slab_volume_km3":np.quantile(slabs,PROBS,axis=1).T.tolist(),"slab_edges_cm":list(LEVELS),
              "power_tail_posterior_fraction":float(full.power_tail.mean()),
              "volume_conditional_on_tail":{k:np.quantile(volume[mask],PROBS).tolist() if mask.any() else None
                  for k,mask in (("power",full.power_tail),("exponential",~full.power_tail))},
              "geographic_score":folds["equal_region_negative_log_score"],"geographic_folds_passed":len(screen["numerically_usable_geographic_folds"]),
              "support_status":screen["status"],"supported_band_policies":contours,
              "fitted_bound_checks":bounds_check,
              "archive_predictions":archive_predictions(path,archives,data,meta,full),
              "world_map":figures(output,name,data,meta,full)}
    if budget:
        log_odds = ODDS_CENTRE+ODDS_SPREAD*full.z[:,3]
        h,L,b,s,p,noise = parameters(full.z)
        from .profile import integrated
        core_share = volume_for(full,np.minimum(L,DIAMETER))/volume
        report["distal_budget"] = {"donor_reference_tail_to_core_odds":DONOR_ODDS,"predictive_df":3,
                    "centre_log_odds":ODDS_CENTRE,"scale_log_odds":ODDS_SPREAD,
                    "reference_log_odds":np.quantile(log_odds,PROBS).tolist(),
                    "physical_distal_volume_fraction":np.quantile(np.clip(1-core_share,0,1),PROBS).tolist()}
    if isinstance(full,Continued):
        variants = {}
        for factor in (.25,1.,4.):
            altered = join(full,data,tilt=factor,donor=full.donor,transfer=full.transfer)
            variants[str(factor)] = np.quantile(altered.volume(),PROBS).tolist()
        report["terminal_continuation"] = {"unrestricted_volume_km3":check["unrestricted_km3"],
             "transfer_multiplier_km3":variants,
             "volume_fraction_beyond_last_sampled_rank":np.quantile(1-volume_for(full,full.edge)/volume,PROBS).tolist(),
             "sampling_rule":"Matched donor remaining/interior mass ratio at the last training-site isopach rank, with one independent factor-two lognormal transfer per deposit."}
    write_json(output/"report.json",report)
    q = report["conditional_volume_km3"]
    lines = [f"# {name} · Bayesian nested whole-area sensitivity","",
             f"**{q[1]:,.1f} km³ of bulk fallout**, with a conditional 95% interval of **{q[0]:,.1f}–{q[2]:,.1f} km³**. Each sample integrates one complete deposit over the Earth.",
             "",f"The {len(data.xy)} accepted rows include {meta['n_measurements']} exact thicknesses, {meta['n_lower_limits']} lower bounds, {meta['n_upper_limits']} upper bounds and {meta.get('n_intervals',0)} two-sided intervals.",
             "", "![Median isopachs and uncertain total area](isopachs.png)", "",
             "Dashed contours show model predictions. Shading in the area panel gives pointwise conditional 95% limits wherever the lower area is positive. [Whole-Earth map](world.png) · [Complete contour support decisions](support.json)",
             "","## Validation","",
             f"The matched geographic score is {report['geographic_score']:.3f}. {report['geographic_folds_passed']}/4 training-fold fits passed the population check. The full-fit population check {'passed' if report['population_check_passed'] else 'retained a warning'}. Direct spatial integration {'passed' if report['area_check'][-1]['passed'] else 'retained a warning'}.",
             "",f"At withheld exact sites, {coverage['covered']}/{coverage['exact_sites']} measurements fell inside their 95% predictive intervals. The median fraction of a draw’s volume inside the retained exact-site triangles is {allocation['fraction_by_part'][0][1]:.1%}. The median fraction beyond the farthest exact site is {allocation['fraction_by_part'][2][1]:.1%}. Geographic classes describe sampling coverage.",
             "",f"{support_note} The global estimate remains conditional on unsampled sectors, the unmeasured core and the distal thinning law.",
             "","## What contributes to volume","",
             "A thickness slab counts the stated slice everywhere the deposit reaches it. The highest slice can occur near or far from the source. Each draw conserves volume across all slices. Marginal medians need not add to the total median.","",
             "| Thickness slice (cm) | Median [conditional 95%], km³ |","|---|---:|"]
    if budget:
        frac = report["distal_budget"]["physical_distal_volume_fraction"]
        lines[4:4] = [f"The refined profile learns the amount deposited beyond its own thinning break. Its median distal share is {frac[1]:.1%} (conditional 95% {frac[0]:.1%}–{frac[2]:.1%}). The thinning break belongs to the deposit and can lie on either side of the outermost measured site.",""]
    if isinstance(full,Continued):
        terminal = report["terminal_continuation"]
        low,med,high = terminal["unrestricted_volume_km3"]
        v = terminal["transfer_multiplier_km3"]
        lines[4:4] = [f"Continuing the fitted slopes freely beyond every sampled isopach rank gives {med:,.1f} km³ [{low:,.1f}–{high:,.1f}]. The whole-area fit transfers the remaining-to-interior mass ratios of four independently mapped falls at the same relative thinning. Reducing or increasing the transferred odds fourfold gives medians of {v['0.25'][1]:,.1f} and {v['4.0'][1]:,.1f} km³. Every accepted training site has exactly the same predicted thickness in these continuations.",""]
        position = lines.index("## Validation")+2
        lines[position:position] = [population_summary(check["numerics"]),""]
        sites = edge_sites(full,data,path)
        sites.to_csv(output/"edge-sites.csv",index=False)
        lines[position:position] = [edge_summary(sites),""]
    labels = [f"{a:g}–{b:g}" for a,b in zip((0.,*LEVELS),LEVELS)]+[f">{LEVELS[-1]:g}"]
    for label,(lo,med,hi) in zip(labels,report["slab_volume_km3"],strict=True):
        lines.append(f"| {label} | {max(0.,med):,.2f} [{max(0.,lo):,.2f}–{max(0.,hi):,.2f}] |")
    lines += ["","## Prior sensitivity","","| Assumption | Median [conditional 95%], km³ | Effective draws |","|---|---:|---:|"]
    for k,s in report["prior_sensitivity"].items():
        lo,med,hi = s["volume_km3"]
        lines.append(f"| {k.replace('_',' ')} | {med:,.1f} [{lo:,.1f}–{hi:,.1f}] | {s['effective_draws']:,.0f} |")
    if bounds_check:
        conflicts = [b for b in bounds_check if b["predictive_probability_of_event"]<.05]
        lines += ["","## Compatibility with fitted bounds","",
                  f"Of {len(bounds_check)} documented thickness bounds, {len(conflicts)} had a posterior-predictive probability below 5% under the fitted model. Each bound entered the likelihood once. This check includes field scatter and known measurement error."]
        if conflicts:
            lines += ["","| Site or source row | Bound (cm) | Median field (cm) | Probability of observed bound |",
                      "|---|---|---:|---:|"]
            for b in conflicts:
                name = b["name"] or f"Row {b['row_id']}"
                limit = (f"{b['lower_cm']:.3g}–{b['upper_cm']:.3g}" if b["event"]=="interval" else
                         (f"≥ {b['lower_cm']:.3g}" if b["event"]=="lower" else f"≤ {b['upper_cm']:.3g}"))
                lines.append(f"| {name} | {limit} | {b['latent_cm'][1]:.3g} | {b['predictive_probability_of_event']:.2%} |")
        lines += ["","The numerical report gives the latent and observation probabilities at every fitted bound. A poorly predicted bound identifies a limitation of the fitted thinning geometry or error model."]
    lines += ["","## Additional distal thinning","",
              "Beyond the largest equivalent isopach radius of any training site in a draw, a smooth extra thinning term tests progressive loss of transported ash. The three fixed scenarios reach an additional e-fold of attenuation after one, two or four more sampled radii. All three integrate to the antipode. The same rule is applied to the training sites in each geographic fold.","",
              "| Additional range | Median [conditional 95%], km³ | Geographic score ↓ |",
              "|---|---:|---:|"]
    for factor,s in distal["scenarios"].items():
        lo,med,hi = s["volume_km3"]
        lines.append(f"| {float(factor):g} × sampled radius | {med:,.1f} [{lo:,.1f}–{hi:,.1f}] | {s['equal_region_negative_log_score']:.3f} |")
    lines += ["","The accepted sites give each scenario the same fitted likelihood. The difference in global volume measures sensitivity to the unobserved continuation. [Quadrature and withheld-site checks](distal-taper.json)"]
    lines += ["","The intervals condition on the recorded sampling locations and the specified shape family. Preservation, reworking and the far-field transfer of analogue slopes can introduce additional uncertainty.",
              "","[Numerical report and archive probes](report.json) · [Withheld thickness predictions](heldout.csv) · [Geographic training IDs](geographic.json) · [Full posterior](full.npz) · [Per-draw thickness budgets](volume-budget.npz)"]
    if isinstance(full,Continued):
        from .bearing_check import review
        review(data,meta,output,name)
        lines += ["","[Matched test of the volume assigned to unsupported bearings](bearing-study/README.md)"]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    return report


def population_summary(numerics):
    populations = np.asarray(numerics["continuation"]["population_volume_quantiles_km3"])
    parent = "passed" if numerics["parent_passed"] else "retained a warning"
    outer = "passed" if numerics["continuation"]["passed"] else "retained a warning"
    return (f"The independent full-data populations gave medians from {min(populations[:,1]):,.1f} to "
            f"{max(populations[:,1]):,.1f} km³. Their upper 95% endpoints ranged from {min(populations[:,2]):,.1f} "
            f"to {max(populations[:,2]):,.1f} km³. The unmodified thinning profile {parent} and the "
            f"continued global volume {outer}. [Population estimates and refinement history](full.json)")


def edge_summary(sites):
    first = sites.iloc[0]
    location = str(first.site).strip() or f"{float(first.latitude):.2f}°, {float(first.longitude):.2f}°"
    return (f"Accepted CSV row {int(first.row_id)} ({location}) sets the last sampled isopach rank "
            f"in {first.edge_probability:.1%} of complete draws and {first.edge_probability_in_largest_5pct_volumes:.1%} "
            f"among the largest 5% by volume. Ties split the credit. This identifies where the shape assumes "
            f"that measured thinning ends. It does not measure the effect of deleting a site. "
            f"[Attribution to every measured or bounded site](edge-sites.csv)")


def nested_dataset(path, output, name=None, *, archives=None, resume=False, parent=None, budget=None, base_cache=None):
    """Fit the principal corridor model, with explicit historical profile options for existing callers."""
    if budget is not None or base_cache is not None:
        if parent is not None:
            raise ValueError("Specify a corridor parent or historical whole-area profile options.")
        return whole_area_dataset(path, output, name, archives=archives, resume=resume,
                                  budget=True if budget is None else budget, base_cache=base_cache)
    from .reach_report import reach_dataset
    return reach_dataset(path, output, name, archives=archives, resume=resume, parent=parent)
