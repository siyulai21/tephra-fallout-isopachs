"""Sensitivity of repeat-site predictions to partial loss of a bed-recognition signal."""
import csv
from itertools import product

import matplotlib.pyplot as plt
import numpy as np


def matching_refits(reports):
    indexed={tuple(sorted((r["scenario"] or {}).get("effective_limit_cm",{}).items())):k for k,r in reports.items()}
    for pair,key in indexed.items():
        if len(pair)==2 and all((x,) in indexed for x in pair):
            yield pair,(indexed[()],indexed[(pair[0],)],indexed[(pair[1],)],key)


def probabilities(evidence,g1,g2):
    # Total probability for the four informative subsets, ordered none, first, second, both.
    fractions=np.array([g1*g2,(1.-g1)*g2,g1*(1.-g2),(1.-g1)*(1.-g2)])
    masses=fractions*np.asarray(evidence)
    return masses/masses.sum()


def publish(output,reports):
    studies=[]
    rows=[]
    for pair,keys in matching_refits(reports):
        outcomes=[reports[k] for k in keys]
        evidence=[np.exp(r["archive_likelihood"]["log_predictive_probability"]) if r["archive_likelihood"]
                  else 1. for r in outcomes]
        ids=[k for k,_ in pair]
        collected={}
        for sid in ids:
            by_case=[next(s for s in r["sites"] if s["id"]==sid) for r in outcomes]
            collected[sid]=(by_case[0]["site"],np.array([[s["probability_local_below"][str(t)]
                                              for t in (.1,1.,10.)] for s in by_case]))
        for g1,g2 in product(np.linspace(0.,1.,21),repeat=2):
            w=probabilities(evidence,g1,g2)
            for sid,(site,values) in collected.items():
                p=w@values
                rows.append({"joint_scenario":keys[-1],"first_archive":ids[0],"second_archive":ids[1],
                    "first_uninformative_probability":g1,"second_uninformative_probability":g2,
                    "probe":site,"p_repeat_local_below_1mm":p[0],"p_repeat_local_below_1cm":p[1],
                    "p_repeat_local_below_10cm":p[2]})
        studies.append((keys[-1],evidence,collected))
    if not rows:
        return []
    with (output/"archive-reliability.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    for key,evidence,s in studies:
        f,axes=plt.subplots(1,2,figsize=(9.,3.65),layout="constrained",sharey=True)
        g=np.linspace(0.,1.,201);w=np.array([probabilities(evidence,v,v) for v in g])
        for ax,(site,vals) in zip(axes,s.values(),strict=True):
            for j,label in enumerate(("1 mm","1 cm","10 cm")):
                ax.plot(g,w@vals[:,j],label=label,lw=2)
            ax.set(xlabel="Assumed chance a site's bed signal is uninformative",xlim=(0,1),ylim=(0,1),title=site)
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Probability repeat local thickness is below level")
        axes[1].legend(frameon=False,fontsize=8)
        f.suptitle("Sensitivity to preservation of a recognizable bed",fontsize=14)
        f.savefig(output/("archive-reliability-"+key+".png"),dpi=170,bbox_inches="tight")
        plt.close(f)
    return [x[0] for x in studies]
