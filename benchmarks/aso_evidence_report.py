"""Figures and matched summaries for the Aso-4 evidence experiments."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmarks.aso_evidence import CASES, dump
from benchmarks.aso_diagnosis import ORIGINAL_GLOBAL
from src.empirical_isopach.models.mass_mixture.inference import flatten
from src.empirical_isopach.models.mass_mixture.geometry import centers, to_sphere, RADIUS, log_field
from src.empirical_isopach.models.mass_mixture.study import audited_data


def matched_validation(reference, alternative):
    assert len(reference["fits"]) == len(alternative["fits"]) == 4
    for key in ("row_ids","folds","buffer_km"):
        assert reference[key] == alternative[key],key
    for a,b in zip(reference["fits"],alternative["fits"]):
        for key in ("training_row_ids","withheld_row_ids"):
            assert a[key] == b[key],key
    return {"reference":reference["metrics"],"alternative":alternative["metrics"],
        "alternative_minus_reference_fold_scores":(np.array(alternative["metrics"]["fold_log_scores"])-reference["metrics"]["fold_log_scores"]).tolist(),
        "all_alternative_fits_passed":alternative["all_fits_passed_diagnostics"],
        "all_reference_fits_passed":reference["all_fits_passed_diagnostics"],
        "interpretation":"Negative differences favour the alternative on matched heldout data. Four overlapping training-fold experiments are not independent external validation or a global-mass accuracy test."}


def load_samples(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def source_map(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    from src.probabilistic_isopach.figures import STYLE, _coastlines
    raw=pd.read_csv("data/eruptions/Aso-4_tephra.csv").iloc[:87]
    revised=pd.read_csv("data/eruptions/Aso-4_reviewed.csv").iloc[:87]
    data,meta=audited_data("data/eruptions/Aso-4_reviewed.csv")
    transform=Transformer.from_crs("EPSG:4326",meta["projection"],always_xy=True)
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(10,7),layout="constrained")
        _coastlines(ax,meta["vent_lonlat"])
        ax.scatter(*data.xy.T,s=15,c="#7B9398",alpha=.7,label="Reviewed observation locations")
        ax.scatter(0,0,marker="*",s=140,c="#B4573F",edgecolors="white",label="Aso source reference",zorder=7)
        for idx,label,shift in ((3,"PC1 · 7 cm",(20,-130)),(6,"PC2 · 6 cm",(100,100))):
            a=np.array(transform.transform(raw.iloc[idx].Lon,raw.iloc[idx].Lat))
            b=np.array(transform.transform(revised.iloc[idx].Lon,revised.iloc[idx].Lat))
            ax.plot(*a,"x",c="#B4573F",ms=9,mew=2)
            ax.plot(*b,"o",c="#087E8B",ms=8)
            ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":"#087E8B","lw":2})
            ax.annotate(label,xy=b,xytext=b+shift,fontsize=10,color="#173B46")
        ax.plot([],[],"x",c="#B4573F",label="Supplied coordinate")
        ax.plot([],[],"o",c="#087E8B",label="Cruise-record coordinate")
        ax.set(xlim=(-500,3100),ylim=(-600,2000),aspect="equal",xlabel="Source-relative easting (km)",ylabel="Northing (km)",
            title="Two source-backed location corrections\nPC1: 30.5° → 34.5° N · PC2: 140° → 146° E · thicknesses unchanged")
        ax.legend(loc="upper left",fontsize=9);ax.grid(alpha=.15)
        fig.savefig(output/"coordinate-corrections.png",dpi=160,bbox_inches="tight");plt.close(fig)


def diagnosis_figures(root,output,original_report,original_global):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.probabilistic_isopach.figures import STYLE, _coastlines
    posterior=Path(original_global)/"posterior.npz"
    assert sha256(posterior.read_bytes()).hexdigest() == original_report["fits"]["baseline"]["posterior_sha256"]
    flat=flatten(load_samples(posterior))
    data,meta=audited_data("data/eruptions/Aso-4_tephra.csv")
    c=centers(flat["log_scale"],flat["offset"])
    sites=to_sphere(data.xy,meta["vent_lonlat"][1])
    nearest=RADIUS*np.arccos(np.clip(np.einsum("dki,ni->dkn",c,sites).max(axis=-1),-1,1))
    distance=np.sum(flat["weights"]*nearest,axis=1)
    volume=np.exp(flat["log_volume"])
    deletion=pd.read_csv(root/"diagnosis/single-row-deletion.csv").sort_values("median_km3")
    selected=pd.concat([deletion.iloc[:5],deletion.iloc[-4:]])
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(12,5.8),layout="constrained")
        x=selected.median_km3.to_numpy();lo=selected.lower95_km3.to_numpy();hi=selected.upper95_km3.to_numpy()
        axes[0].errorbar(x,np.arange(len(x)),xerr=[x-lo,hi-x],fmt="o",color="#187E86",capsize=3)
        axes[0].axvline(original_report["fits"]["baseline"]["volume"]["median"],color="#B4573F",ls="--",label="All 87 records")
        axes[0].set(yticks=np.arange(len(x)),yticklabels=[f"Omit row {r.csv_row}: {r.thickness_cm:g} cm" for r in selected.itertuples()],
            xscale="log",xlabel="Bulk fallout (km³); conditional 95% interval",title="A · Single-record influence\nPSIS deletion approximation")
        axes[0].invert_yaxis();axes[0].legend(loc="lower right",fontsize=9)
        axes[1].hexbin(distance,volume,xscale="log",yscale="log",gridsize=45,mincnt=1,cmap="Blues",bins="log")
        axes[1].set(xlabel="Mass-weighted centre-to-nearest-site distance (km)",ylabel="Posterior total bulk fallout (km³)",
            title=f"B · Unmeasured peaks and high volume\nSpearman correlation {spearmanr(distance,volume).statistic:.3f}")
        fig.suptitle("Aso-4: diagnose the high tail without deleting inconvenient data",fontsize=14)
        fig.savefig(output/"influence-and-geometry.png",dpi=160);plt.close(fig)
        # Pointwise median fields conditioned on low/high total-volume quartiles.
        # These illustrate posterior dependence, not competing fitted deposits.
        x=np.linspace(-3000,4500,151);y=np.linspace(-3000,4500,151)
        xx,yy=np.meshgrid(x,y);points=np.column_stack([xx.ravel(),yy.ravel()])
        unit=to_sphere(points,meta["vent_lonlat"][1]); levels=np.log([.1,1,3,10,30,100,300,1000])
        fig,axes=plt.subplots(1,2,figsize=(12,6.4),layout="constrained")
        panel_records=[]
        for ax,label,mask in zip(axes,("Lowest-volume quarter","Highest-volume quarter"),
                (volume<=np.quantile(volume,.25),volume>=np.quantile(volume,.75))):
            idx=np.flatnonzero(mask);idx=idx[np.linspace(0,len(idx)-1,192).astype(int)]
            subset={k:v[idx] for k,v in flat.items()}
            z=np.concatenate([np.median(log_field(subset,unit[j:j+256]),axis=0) for j in range(0,len(unit),256)]).reshape(xx.shape)
            fill=ax.contourf(xx,yy,z,levels=levels,cmap="YlOrRd",extend="both",alpha=.75)
            _coastlines(ax,meta["vent_lonlat"])
            cs=ax.contour(xx,yy,z,levels=np.log([1,10,100,1000]),colors="#75352D",linewidths=.7,linestyles="dashed")
            ax.clabel(cs,fmt=lambda v:f"{np.exp(v):g}",fontsize=8)
            ax.scatter(*data.xy.T,s=10,c="white",edgecolors="#183843",linewidths=.4,zorder=6)
            ax.scatter(0,0,marker="*",s=140,c="#0D607D",edgecolors="white",zorder=7)
            v=np.quantile(volume[mask],[.025,.5,.975])
            ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect="equal",xlabel="Source-relative easting (km)",ylabel="Northing (km)",
                title=f"{label}\nWithin-group median total {v[1]:,.0f} km³")
            panel_records.append({"group":label,"selected_draw_indices":idx.tolist(),"conditional_group_volume_quantiles":v.tolist()})
        cb=fig.colorbar(fill,ax=axes,shrink=.75,pad=.02,ticks=levels)
        cb.ax.set_yticklabels(["0.1","1","3","10","30","100","300","1,000"]);cb.set_label("Conditional median thickness (cm)")
        fig.suptitle("Same original posterior, split by total volume\nDots: observations · star: volcano · map extent never sets volume",fontsize=13)
        fig.savefig(output/"low-high-volume-fields.png",dpi=160);plt.close(fig)
        dump(output/"map-design.json",{"groups":panel_records,"grid_km":{"x":[-3000,4500,151],"y":[-3000,4500,151]},
            "scope":"192 deterministically spaced posterior fields per group; map is pointwise log-thickness median. Its integral is not the posterior median volume. Dashed contours are model summaries, not certified supported isopachs."})


def summarize(root,output,original_global=ORIGINAL_GLOBAL):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    original=json.loads((Path(original_global)/"report.json").read_text())
    results={name:json.loads((root/"study"/name/"report.json").read_text()) for name in CASES}
    revised=json.loads((root/"revision/baseline/study.json").read_text())
    revised_offset=json.loads((root/"revision/offset_sd1/study.json").read_text())
    coordinate=json.loads((root/"revision/coordinate_only/report.json").read_text())
    marine=json.loads((root/"marine/report.json").read_text())
    conflict=json.loads((root/"conflict/report.json").read_text())
    assert set(marine["results"]) == {"offset_sd3", "offset_sd1"}
    assert set(conflict["results"]) == {"primary_6p5", "primary_7"}
    row=lambda key,fit,scope:{"case":key,"scope":scope,**{k:fit["volume"][k] for k in ("lower95","median","upper95")},
        "sampling_passed":fit["diagnostics"]["passed"],"attempts":len(fit["attempts"]),"posterior_sha256":fit["posterior_sha256"]}
    rows=[row("original",original["fits"]["baseline"],"Original 87 exact records; displacement SD 3")]
    rows += [row(name,r["fit"],CASES[name]) for name,r in results.items()]
    rows += [row("coordinates_only",coordinate["fit"],"Only two cruise-coordinate corrections"),
        row("reviewed",revised["fits"]["baseline"],"Coordinate corrections and four source-qualified bounds; displacement SD 3"),
        row("reviewed_offset_sd1",revised_offset["fits"]["baseline"],"Same reviewed data; displacement SD 1")]
    rows += [row("reviewed_"+k,v["fit"],"Corrected coordinates and three other bounds; Biwa primary-table exact endpoint")
             for k,v in conflict["results"].items()]
    rows += [row("omit_marine_"+k,v["fit"],"Reviewed input, omit all 28 S2 submarine records; 59 retained (not certified land-only)")
             for k,v in marine["results"].items()]
    pd.DataFrame(rows).to_csv(output/"experiments.csv",index=False)
    summary={"experiments":rows,"original_data_validation":matched_validation(original["validation"],results["offset_sd1"]["validation"]),
        "reviewed_data_validation":matched_validation(revised["validation"],revised_offset["validation"]),
        "marine_setting_test":{"protocol":marine["protocol"],"withheld_metrics":{k:v["withheld_metrics"] for k,v in marine["results"].items()}},
        "source_conflict_test":conflict["protocol"],
        "interpretation":"Separate conditional credible intervals. The collection of alternatives has no probability weights and is not a joint confidence region. Source revisions, deletions and geometry priors are not interchangeable evidence.",
        "builder_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(output/"summary.json",summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.probabilistic_isopach.figures import STYLE
    labels={"original":"Original input / SD 3","bounds_only":"Three recovered bounds","s2_bounds":"Plus S2 lower bound at Biwa",
        "primary_6p5":"Biwa endpoint 6.5 cm","primary_7":"Biwa endpoint 7 cm","omit_ambiguous":"Omit 4 uncertain proximal sites",
        "omit_row37":"Omit offshore row 37","omit_row49":"Omit offshore row 49","offset_sd1":"Original input / SD 1",
        "offset_sd6":"Original input / SD 6","coordinates_only":"Only correct two coordinates","reviewed":"Coordinates + four bounds / SD 3",
        "reviewed_offset_sd1":"Coordinates + four bounds / SD 1",
        "reviewed_primary_6p5":"Corrected coordinates / Biwa 6.5 cm",
        "reviewed_primary_7":"Corrected coordinates / Biwa 7 cm",
        "omit_marine_offset_sd3":"Omit 28 compiled marine sites / SD 3",
        "omit_marine_offset_sd1":"Omit 28 compiled marine sites / SD 1"}
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(11,9.5),layout="constrained")
        for i,r in enumerate(rows):
            color="#A86627" if "offset" in r["case"] else "#087E8B"
            ax.plot([r["lower95"],r["upper95"]],[i,i],color=color,lw=2)
            ax.scatter(r["median"],i,c=color,marker="o" if r["sampling_passed"] else "x")
        ax.set(xscale="log",yticks=np.arange(len(rows)),yticklabels=[labels[r["case"]] for r in rows],
            xlabel="Whole-Earth bulk fallout (km³; log scale)",title="Aso-4: source and geometry experiments\nPoints: medians · lines: separate conditional 95% credible intervals")
        ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
        fig.savefig(output/"experiment-intervals.png",dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
        for ax,key,title in zip(axes,("original_data_validation","reviewed_data_validation"),("Original data","Corrected locations + source bounds")):
            group=summary[key]
            for name,label,color in (("reference","Displacement SD 3","#087E8B"),("alternative","Displacement SD 1","#A86627")):
                ax.plot(np.arange(1,5),group[name]["fold_log_scores"],"o-",label=label,color=color)
            ax.set(xticks=range(1,5),xlabel="Withheld geographic region",ylabel="Negative log predictive score (lower is better)",title=title)
            ax.legend(fontsize=9)
        fig.suptitle("Matched geographic prediction · separate evaluation for each data version",fontsize=13)
        fig.savefig(output/"geographic-prediction.png",dpi=160);plt.close(fig)
    source_map(output)
    diagnosis_figures(root,output,original,original_global)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--original-global",type=Path,default=ORIGINAL_GLOBAL)
    args=parser.parse_args();summarize(args.root,args.output,args.original_global)
