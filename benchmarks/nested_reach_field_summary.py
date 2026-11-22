"""Publish observations, spatially separated scores and empirical contour comparisons."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from .nested_reach_field import FIELD_CASES, partition
from src.empirical_isopach.core.validation import spatial_folds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field",type=Path,required=True)
    a=parser.parse_args()
    root=a.field
    rows=[]
    for slug in FIELD_CASES:
        home=root/slug
        v=json.loads((home/"validation.json").read_text())
        train,test,meta,path=partition(slug,home)
        z=v["tested_models"]
        scores=np.array([z[m]["row_log_score"] for m in ("whole_area","corridor")])
        labels=spatial_folds(test.xy,4) if len(test.xy)>=8 else np.arange(len(test.xy))
        drop=[float(np.mean(scores[0,labels!=k]-scores[1,labels!=k])) for k in np.unique(labels)]
        separated=v["geographically_separated"]
        rows.append({"case":slug,"training_rows":len(train.xy),"withheld_rows":len(test.xy),
                     "old_score":float(-scores[0].mean()),"new_score":float(-scores[1].mean()),
                     "separated_test_rows":len(separated["test_rows"]),"buffer_km":separated["training_only_buffer_km"],
                     "separated_old":separated["scores"]["whole_area"],"separated_new":separated["scores"]["corridor"],
                     "leave_one_test_block_out_change_min":min(drop),"leave_one_test_block_out_change_max":max(drop),
                     "reference_numerics_passed":v["numerics"]["passed"],
                     "revised_numerics_passed":v["revised_source_passed"] and v["importance"][-1]["passed"]})
        table=pd.DataFrame({"accepted_csv_row":test.row_ids,"observed_or_bound_cm":np.exp(test.log_value),
                            "censored":test.censored,"lower_bound":test.lower_censored,
                            "nearest_training_km":v["nearest_training_projected_km"],"test_block":labels,
                            "old_log_score":scores[0],"new_log_score":scores[1],
                            "old_latent_median_cm":np.exp(z["whole_area"]["latent_log_cm"]),
                            "new_latent_median_cm":np.exp(z["corridor"]["latent_log_cm"])})
        table.to_csv(home/"held-survey.csv",index=False,lineterminator="\n")
    pd.DataFrame(rows).to_csv(root/"survey-scores.csv",index=False,lineterminator="\n")
    markup=["# Tests against real fallout deposits","",
        "Each fit below omits the survey or archive it predicts. The same reach assumptions were frozen before the external Calbuco test. The checked quantities are site thickness, documented bounds and mapped contour intersections. No eruption has a direct measurement of its complete deposited volume.","",
        "| Training evidence | Withheld evidence | Sites or bounds tested | Whole-area score | Corridor score |",
        "|---|---|---:|---:|---:|"]
    names=[("Calbuco · Van Eaton survey","Reckziegel survey"),("Aso-4 · verified subaerial settings","verified submarine cores"),
           ("B-Tm · all other accepted rows","three interpreted archive limits"),("Pinatubo · land sections and sublayer maps","Wiesner–Wang marine cores")]
    for r,(name,test) in zip(rows,names,strict=True):
        markup.append(f"| {name} | [{test}]({r['case']}/held-survey.csv) | {r['withheld_rows']} | {r['old_score']:.3f} | {r['new_score']:.3f} |")
    c=json.loads((root/"calbuco-2015/validation.json").read_text())
    contour=json.loads((root/"calbuco-2015/mapped-01mm.json").read_text())
    v=c["tested_models"]
    lo,med,hi=c["corridor_integration"]["volume_km3"]
    old=c["whole_area_volume_km3"]
    r=rows[0]
    markup += ["","Lower scores indicate better held-survey predictions. The score averages the negative log predictive density of an exact measurement or the negative log probability of a reported bound. Compare models within a row. Pinatubo’s land-only fit retained a numerical warning. "
               "[Scores, spatial buffers, block deletion checks and numerical flags](survey-scores.csv)","",
               "## Calbuco 2015 · A separate eruption and independent surveys","",
               "The training likelihood uses 100 exact and 59 interval readings from the Van Eaton compilation. Four reported zeros lack a site-specific detection limit. The independent Reckziegel file contains 45 rows. One reports zero and two share both coordinates and the same thickness. The 43 distinct positive readings supply the quantitative test. Each source row and exclusion is traceable. "
               "[Original archive](../../../data/sources/nested-reach/README.md) · [Source-row decisions](calbuco-2015/row-decisions.csv) · [Accepted observations](calbuco-2015/field-observations.csv)","",
               f"On all 43 test sites the score changes from {r['old_score']:.3f} to {r['new_score']:.3f}. The original and revised 95% predictive intervals contained {v['whole_area']['covered95_exact']} and {v['corridor']['covered95_exact']} exact readings. "
               f"At the {r['separated_test_rows']} sites farther than the training-derived {r['buffer_km']:.2f} km buffer from every training observation, the score changes from {r['separated_old']:.3f} to {r['separated_new']:.3f}. "
               f"Leaving out each geographic test block in turn gives a new-minus-old score change from {r['leave_one_test_block_out_change_min']:+.3f} to {r['leave_one_test_block_out_change_max']:+.3f}. The independent campaign contains closely spaced sites, and this is one eruption. "
               "[Per-site predictions](calbuco-2015/held-survey.csv) · [Measured threshold crossings](calbuco-2015/validation.json)","",
               "![Calbuco distal isopach and model disagreement along ten real-map transects](calbuco-2015/mapped-contour.png)","",
               f"Romero and colleagues’ field reconstruction provides a separate map comparison. At ten parallels from 36.5° to 41°S, the mean extra length of the modeled 0.1 mm footprint falls from {contour['whole_area']['extra']:.1f} to {contour['corridor']['extra']:.1f} km. "
               f"The mean missed mapped length changes from {contour['whole_area']['missed']:.1f} to {contour['corridor']['missed']:.1f} km. The reference line is an expert interpretation between field sites. "
               "The magenta rings give the digitized crossings. The sampling campaign and the reference line stayed outside the fit. "
               "[Printed figure and attribution](../../../data/sources/nested-reach/README.md) · [Transect measurements](calbuco-2015/mapped-01mm-transects.csv) · [Mingari et al., Figure 2](https://gmd.copernicus.org/articles/16/3459/2023/)","",
               f"The Van Eaton training survey gives {old[1]:.3f} km³ [{old[0]:.3f}–{old[2]:.3f}] under whole-area continuation and {med:.3f} km³ [{lo:.3f}–{hi:.3f}] under the principal corridor model. "
               "Published field reconstructions span approximately 0.27–0.58 km³ bulk. One of those reconstructions uses the Van Eaton observations. Romero et al.’s Table 3 gives 0.270 km³ for the whole deposit under two-segment exponential thinning. These estimates provide geological context and have their own distal extrapolation assumptions. "
               "[Review of the field estimates](https://gmd.copernicus.org/articles/16/3459/2023/) · [Romero et al., original reconstruction](https://doi.org/10.1016/j.jvolgeores.2016.02.027)","",
               "## What the named eruptions establish","",
               "At B-Tm, all three interpreted limits were omitted together. The probability of satisfying the Sihailongwan upper limit rises from 0.18% to 97.85%. These archives helped motivate the revision, so this is a retrospective check. The lake supplies an inferred upper bound on equivalent bed thickness. Calbuco supplies the external survey test. [B-Tm predictions by archive](b-tm/held-survey.csv)","",
               "At Aso-4 the 22 verified terrestrial rows predict the thicknesses of 28 verified submarine cores. Rows with unknown settings and lake cores were excluded from this transfer experiment. The whole-area sensitivity predicts the withheld marine thicknesses better. The survey provides measurements chiefly east and northeast of the vent. It leaves the western volume unresolved. [Aso-4 marine predictions](aso-4/held-survey.csv)","",
               "For Pinatubo, 127 accepted land sites and bounds predict ten withheld marine-core readings, including seven exact thicknesses. The corridor model underpredicts offshore reach. Neither model captures the offshore survey adequately and the land-only fit retained a sampling warning. This is direct evidence that a sampling boundary can truncate real distal fallout. [Marine-core predictions](pinatubo-1991/held-survey.csv) · [Original marine survey](https://pubs.usgs.gov/pinatubo/wiesner/index.html)","",
               "Mazama’s supplied thickness table lacks survey identifiers. Its evidence comes from buffered geographic and compass holdouts of actual measurements. The southwestern sector has no exact thicknesses or bounds, so the modeled southwestern closure remains conditional. [Mazama maps and spatial checks](../../../eruptions/mazama/results/nested-isopachs/README.md)"]
    (root/"README.md").write_text("\n".join(markup)+"\n")


if __name__=="__main__":
    main()
