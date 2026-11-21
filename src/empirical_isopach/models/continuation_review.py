"""Matched predictions and maps for two optional global-family prior sensitivities."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.audit import audited_data
from ..visualization.maps import frame_bounds, grid_axes, observations, observation_legend, setup_map
from .directional.azimuth.primary import read_field
from .directional.ensemble.fields import inverse_cdf
from .directional.ensemble.members import splits
from .directional.gap_amount import FACTORS, PRIOR, holdouts, integrated_amount, weight_summary
from .mass_mixture.bearings.field import BearingField, BearingSupport, phases, selected
from .mass_mixture.displacement import heldout, prior_ratio
from .nested.study import digest, write_json

PROBS = (.025,.5,.975)


def read(path):
    return json.loads(path.read_text())


def checked(base, relative):
    if digest(base/relative)!=read(base/"artifacts.json")[relative]:
        raise ValueError(f"Published field digest differs for {relative}")
    return np.load(base/relative,allow_pickle=False)


def paired_map(folder, data, meta, title, evaluator, labels):
    frame = frame_bounds(data.xy)
    x,y = grid_axes(frame,longest=145,minimum=65)
    xx,yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(),yy.ravel()))
    values = np.concatenate([evaluator(batch) for batch in np.array_split(xy,(len(xy)+95)//96)],axis=1)
    fig,axes = plt.subplots(1,2,figsize=(12,5.6),layout="constrained")
    for ax,z,label in zip(axes,values,labels,strict=True):
        z = z.reshape(xx.shape)/np.log(10.)
        image = ax.pcolormesh(x,y,z,cmap="viridis",vmin=-1,vmax=max(2.,np.ceil(values.max()/np.log(10.))),
                              shading="auto",rasterized=True)
        levels = [v for v in (-1,0,1,2,3) if z.min()<v<z.max()]
        if levels:
            line = ax.contour(x,y,z,levels=levels,colors="white",linewidths=.9,linestyles="dashed")
            ax.clabel(line,fmt=lambda v:f"{10**v:g}",fontsize=8)
        setup_map(ax,frame,meta["vent_lonlat"])
        observations(ax,data,meta=meta,size=13)
        ax.set_title(label)
    fig.colorbar(image,ax=axes,label="Median log₁₀ thickness (cm)",shrink=.76)
    fig.suptitle(title)
    observation_legend(fig,data,meta=meta)
    fig.savefig(folder/"isopachs.png",dpi=135)
    plt.close(fig)


def emit(folder,result,note):
    valid = [f for f in result["folds"] if f.get("status","estimated")=="estimated"]
    result["paired_scores"] = {k:float(np.mean([f[k] for f in valid])) for k in
                               ("reference_score","sensitivity_score")}
    write_json(folder/"report.json",result)
    q = result["sensitivity_volume_km3"]
    scores = result["paired_scores"]
    lines = [f"# {result['name']} · {result['label']}", "",
             f"The candidate median complete-deposit volume is **{q[1]:,.1f} km³**. The conditional 95% interval is {q[0]:,.1f}–{q[2]:,.1f} km³.", "",
             note, "", f"Matched buffered geographic scores averaged {scores['reference_score']:.3f} under the reference and {scores['sensitivity_score']:.3f} under the candidate. Smaller values indicate better site predictions. Both scores use the same posterior subset within each fold.", "",
             f"Numerical and parent sampling checks together **{'passed' if result['passed'] else 'retained warnings'}**. The output includes every attempted fold and its parent status.", "",
             "![Matched median isopachs](isopachs.png)", "",
             "Dashed lines show model contours in the survey window. Both global volumes use the complete Earth. [Posterior checks, assumptions and per-site scores](report.json)"]
    (folder/"README.md").write_text("\n".join(lines)+"\n")


def directional_review(path, base, output, work):
    data,meta = audited_data(path)
    base,folder = Path(base),Path(output)
    folder.mkdir(exist_ok=True,parents=True)
    ref = read(base/"azimuth/report.json")
    if ref["input"]["input_sha256"]!=meta["input_sha256"]:
        raise ValueError("Directional case and CSV differ")
    model = read_field(base/"azimuth",data,ref)
    mass,history = integrated_amount(model)
    original = ref["volume"]["coherent_volume_km3"]
    expected = [original[k] for k in ("lower95","median","upper95")]
    recovery = float(np.max(abs(inverse_cdf(mass[1],model.weights,PROBS)/expected-1)))
    folds = holdouts(data,meta["vent_lonlat"][1],work,ref["weights"],ref)
    def evaluate(xy):
        _,fields,bary = model.components(xy)
        h,_ = model.support.evaluate(xy,bary)
        # Deterministic stratification in posterior mass limits map memory.
        idx = np.searchsorted(model.weights.cumsum(),(np.arange(384)+.5)/384)
        f = fields[idx]
        shifted = f[None,:,:]+np.log(FACTORS)[:,None,None]*h
        mixed = inverse_cdf(shifted.reshape(-1,len(xy)).T,np.repeat(PRIOR/384,384))[0]
        return np.stack((np.median(f,axis=0),mixed))
    paired_map(folder,data,meta,ref["dataset"],evaluate,("Uncertain bearing","Uncertain bearing and gap amount"))
    fin = history[-1]
    result = {"name":ref["dataset"],"label":"Uncertain amount of fallout in unsampled bearings",
              "input_sha256":meta["input_sha256"],"reference_packet_sha256":digest(base/"azimuth/fields.npz"),
              "factors":FACTORS.tolist(),"prior_probabilities":PRIOR.tolist(),"reference_volume_km3":expected,
              "sensitivity_volume_km3":weight_summary(mass,model.weights).tolist(),
              "volume_by_factor_km3":[inverse_cdf(v,model.weights,PROBS).tolist() for v in mass],
              "integration":history,"reference_recovery_relative":recovery,"folds":folds,
              "passed":bool(recovery<.05 and fin["doubling_change"] is not None and
                            max(fin["scramble_change"],fin["doubling_change"])<.025
                            and all(m["diagnostics"]["passed"] for m in ref["members"].values())
                            and all(f.get("parent_passed",False) for f in folds))}
    emit(folder,result,"One amount multiplier is shared by the entire deposit. Its influence increases with the existing measure of unsupported bearing and vanishes at training sites. The quarter, unchanged and fourfold factors are structural scenarios. An unvisited bearing supplies no measurement of that amount.")
    return result


def spherical_review(path, base, output):
    data,meta = audited_data(path)
    base,folder = Path(base),Path(output)
    folder.mkdir(exist_ok=True,parents=True)
    ref = read(base/"report.json")
    if ref["input"]["input_sha256"]!=meta["input_sha256"]:
        raise ValueError("Spherical case and CSV differ")
    with checked(base,"posterior.npz") as z:
        posterior = dict(z)
    with checked(base,"bearing/baseline.npz") as b:
        mass = b["volume_km3"]
        ratio = prior_ratio(posterior).ravel()[b["flat_index"]]
        chain = b["flat_index"]//posterior["log_volume"].shape[1]
    w = np.repeat(ratio,3)
    reference_q = np.quantile(mass,PROBS)
    quantiles = reference_q if np.all(ratio==1) else inverse_cdf(mass.ravel(),w,PROBS)
    ess = float(ratio.sum()**2/(ratio@ratio))
    chain_medians = [float(inverse_cdf(mass[chain==c].ravel(),np.repeat(ratio[chain==c],3))[0]) for c in np.unique(chain)]
    lat = meta["vent_lonlat"][1]
    files = list((base/"support").glob("*.npz"))
    if len(files)!=1:
        raise ValueError("Expected one verified spherical fold packet")
    folds = []
    with checked(base,str(files[0].relative_to(base))) as packet:
        for i,train,test in splits(data,lat,"geographic"):
            record = ref["validation"]["fits"][i]
            if record["training_row_ids"]!=train.row_ids.tolist() or record["withheld_row_ids"]!=test.row_ids.tolist():
                raise ValueError("Spherical geographic rows differ")
            prefix = f"fold-{i}__"
            draws = {k[len(prefix):]:packet[k] for k in packet if k.startswith(prefix)}
            chains = len(record["fit"]["diagnostics"]["energy_bfmi_per_chain"])
            old,new,fold_ess = heldout(draws,train,test,lat,chains)
            folds.append({"fold":i,"training_rows":train.row_ids.tolist(),"test_rows":test.row_ids.tolist(),
                          "reference_score":float(-old.mean()),"sensitivity_score":float(-new.mean()),
                          "per_site_reference":old.tolist(),"per_site_sensitivity":new.tolist(),"effective_draws":fold_ess,
                          "parent_passed":record["fit"]["diagnostics"]["passed"]})
    draws,_,shape = selected(posterior,64)
    field = BearingField(draws,BearingSupport(data,lat),phases(shape))
    map_weights = np.tile(prior_ratio(draws),3)
    def evaluate(xy):
        fields = np.concatenate(field.components(xy))
        baseline = np.median(fields,axis=0)
        revised = baseline if np.all(map_weights==1) else inverse_cdf(fields.T,map_weights)[0]
        return np.stack((baseline,revised))
    paired_map(folder,data,meta,ref["name"],evaluate,("Published bearing prior","Soft source displacement prior"))
    result = {"name":ref["name"],"label":"Soft source displacement prior","input_sha256":meta["input_sha256"],
              "reference_volume_km3":reference_q.tolist(),"sensitivity_volume_km3":quantiles.tolist(),
              "parent_sha256":digest(base/"posterior.npz"),"fold_packet_sha256":digest(files[0]),
              "effective_parent_draws":ess,"parent_draws":len(ratio),"chain_medians_km3":chain_medians,"folds":folds,
              "passed":bool(ess>100 and min(chain_medians)/max(chain_medians)>.75
                            and ref["fits"]["baseline"]["diagnostics"]["passed"] and
                            all(f["parent_passed"] and f["effective_draws"]>=32 for f in folds))}
    emit(folder,result,"The prior favors component centres within several component widths of the source. The penalty is smooth and retains positive probability for detached maxima. Source-focused kernels receive unit weight. Effective draw counts diagnose the importance reweighting.")
    return result
