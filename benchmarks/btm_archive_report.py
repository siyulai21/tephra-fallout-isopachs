"""Rebuild the field interpretation for the B-Tm western archive probes."""
import json
from pathlib import Path

import numpy as np

from src.empirical_isopach.models.directional.study import digest,write_json


def read(path):
    return json.loads(Path(path).read_text())


def format_cm(value):
    if value < .001:
        return f"{value:.2g}"
    return f"{value:.3g}"


def publish(output):
    output=Path(output)
    scenarios={f.parent.name:read(f) for f in output.glob("*/result.json")}
    original=read(output.parent/"ensemble/report.json")
    sphere=read(output/"spherical-archive-checks.json")["scenarios"]
    base=scenarios["quantitative"];ref=scenarios["visible_analogs"]
    old_sites=read(output/"published-map-at-archives.json")
    select=lambda r,s:next(x for x in r["sites"] if x["id"]==s)
    def west(d):return float(np.asarray(d["median_surface"]["parts_sectors_km3"])[:,[5,6,7]].sum())
    rows=["# What the western B-Tm cores say about the isopachs","",
          "The occurrence of Millennium ash is secure at Lake Sihailongwan and Quanyang peat. Both lie beyond the regional observation hull and outside every measured interpolation triangle. Neither site supplies a primary fallout thickness. Their sediments do test whether a proposed thick layer could plausibly have left the reported trace.","",
          "| Observation | What was actually recovered |",
          "|---|---|",
          "| Lake Sihailongwan, 124 km WNW | Successive 1 cm samples contain distinct colourless and brown glass peaks, 2,657 and 669 shards per 0.5 g dry sediment. The varved core preserved unabraded glass. The event is recorded as cryptotephra. The authors also identified a visible, older black scoria horizon at 78–79 cm core depth. [Sun et al., sections 2–4 and Fig. 2](https://doi.org/10.1016/j.gr.2015.01.013) |",
          "| Quanyang, 46 km NW | In a continuous 1 cm peat sequence, the 64 cm slice contained 20–25 dispersed pumice clasts/cm³. No continuous bed was seen. Adjacent slices lacked large pumice. The largest clasts reached about 3 mm. The fine fraction and the relation between peat retention and original fall remain unknown. [Han et al., section 1.1 and Fig. 1](https://www.dzdz.ac.cn/article/2019/0253-4967-41-1-225.html) |",
          "| Lake Kushu, east of the vent | An independently retained occurrence test in all western refits. The lake core contained an invisible B-Tm peak of 30,671 shards/g. A local fallout detection limit is unavailable. [Chen et al., 2016](https://doi.org/10.1016/j.quageo.2016.02.003) · [Chen et al., 2019](https://doi.org/10.1016/j.quascirev.2019.05.017) |",
          "| Gushantun peat | The reported horizon is much younger than 946 CE. It remains an attribution dispute. It supplies no evidence for thin or absent Millennium fallout. [Source-row audit](../../../../../data/audits/B-Tm/evidence.csv) |","",
          "The 1 cm slice is the vertical support of a core sample. Loss, sediment focusing, compaction and inclusion of the fine fraction affect the transfer to bulk fallout. A recent Quanyang peat study measured density in shallow cores with basal ages about 1800 and 1875 CE. Those cores do not reach the Millennium horizon. [Quanyang coring and chronology](https://doi.org/10.11928/j.issn.1001-7410.2025.03.02)","",
          "## Predictions at the actual cores","",
          "| Median latent primary thickness (cm) | Sihailongwan | Quanyang | Kushu, held out |",
          "|---|---:|---:|---:|"]
    ids=("longgang","quanyang","old-084")
    rows.append("| Previous six-structure **published map** | "+" | ".join(format_cm(old_sites[k]["map_field_cm"]) for k in ids)+" |")
    for title,source in [("Spherical thickness-only",sphere["quantitative"]),("Measured fan directions, thickness-only",base),
                         ("Directional, comparable visible beds",ref),("Spherical, comparable visible beds",sphere["visible_analogs"]),
                         ("Directional, ten-slice effective scale",scenarios["both_10"]),
                         ("Spherical, ten-slice effective scale",sphere["both_10"]),
                         ("Directional, condition on Sihailongwan alone",scenarios["lake_only"]),
                         ("Directional, condition on Quanyang alone",scenarios["peat_only"])]:
        rows.append(f"| {title} | "+" | ".join(format_cm(select(source,k)["latent_cm"]["median"]) for k in ids)+" |")
    rows+=["","The previous six-structure map assigns "+f"{west(original):.2f} km³ to SW, W and NW combined. Restricting the central structures to measured fans and allowing shorter cross-bearing reach gives {west(base):.2f} km³. The comparable-bed scenario gives {west(ref):.2f} km³. The same 68 quantitative observations enter all three. Ten-slice, one-slice and other archive assumptions remain separate sensitivities. Their scenario membership, weights and numerical warnings are visible in the main report.","",
           "The lake and peat limits in the comparable-bed sensitivity come from identifiable B-Tm in Lake Ogawara (1 cm) and a peat-bog margin in Primorye (up to 2 cm). They test an assumption about equivalent preservation and visual recognition at the two western sites. The Primorye investigators recovered only cryptotephra at the centre of that same bog, direct evidence that local preservation can change the result. [Ogawara, Figs. 1c and 2](https://doi.org/10.2343/geochemj.2.0619) · [Razjigaeva et al., ash-layer distributions](https://sciencejournals.ru/view-article/?a=DANZem2010014Razzhigaeva&j=danzem&n=2&v=494&y=2020)","",
           "An upper-only fit can also make the western lake extremely thin. Sihailongwan preserved abundant glass in the extraction fraction. The smallest fitted thicknesses require an explicit check against shard loading before a physical interpretation. Costa’s published Sihailongwan equivalent is 0.00256 cm under a transferred shard-count calibration. The necessary site-specific conversion from the counted fraction to original fallout remains unresolved. The transfer alternatives also disagree at independent Lake Kushu. These facts limit any claim of a uniquely located western isopach. [Reported equivalents and count audit](../../../../../data/audits/B-Tm/README.md)","",
           "## What the field evidence can reject","",
           "The profiles challenge an intact, continuous decimetre bed at these locations. Assigning a sharp upper bound to the original airfall additionally requires a defensible archive response. The table below tests the two unfitted predictions at 1 cm. The probability uses local variability and measurement-model scatter. A low value measures tension conditional on 1 cm effective recognition, even if the median field itself is smooth.","",
           "| Model before archive conditioning | P(local thickness below 1 cm), Sihailongwan | P(local thickness below 1 cm), Quanyang |",
           "|---|---:|---:|"]
    for title,entries,field in (("Six-structure directional",old_sites,None),("Measured fan directions",base,"probability_local_below"),
                                ("Spherical mixture",sphere["quantitative"],"p_local_below_scale")):
        ps=([entries[k]["prob_local_under_1mm_or_1cm"][1] for k in ids[:2]] if field is None else
            [select(entries,k)[field]["1.0"] for k in ids[:2]])
        rows.append(f"| {title} | "+" | ".join(f"{value:.2%}" for value in ps)+" |")
    geo=read(output/"visible_analogs/validation.json")
    gs=geo["geographic"]["equal_region_negative_log_score"]
    qs=geo["quadrants"]["equal_region_negative_log_score"]
    flags=[str(f["fold"]) for f in geo["geographic"]["folds"] if f["status"]=="estimated" and not f["passed"]]
    rows+=["","As a reverse calculation, if 1 cm of visible-equivalent material could be recognized, the published local median fields require net visible-equivalent retention below "
           f"{1/old_sites['longgang']['map_field_cm']:.1%} at Sihailongwan and {1/old_sites['quanyang']['map_field_cm']:.1%} at Quanyang to evade recognition. Those fractions are model implications, with no assigned field probability.","",
           f"The comparable-bed directional fit scored {gs:.3f} across buffered geographic groups and {qs:.3f} across estimable quadrants. Archives were removed along with the held-out geography and its buffer. Geographic folds with numerical warnings: {', '.join(flags) if flags else 'none'}. A positive-only thickness holdout can favour a broad fan. The independent six-eruption comparison found a predictive cost from applying angular narrowing everywhere. [Matched calibration and all six holdout comparisons](../../../../../comparisons/bearing-continuation/README.md)","",
           "The regional GP and spline integrations end at the observation hull. Their published totals supply no western-site prediction at these four coordinates. The independently normalized spherical family can extend west and has its own archive likelihood and conditional volume interval. The maps mark where the directional scenarios disagree and keep all-quantitative and hypothetical-fan estimates readable. [Median isopachs, volumes and uncertainty](README.md) · [Reusable archive protocol](../../../../../docs/methods/archive-evidence.md)",""]
    rows+=["## Independent marine occurrences","",
           "The B-Tm glass peak in Ulleung Basin core 13PT-P4_026 contains 9,529 shards/g in the 25–27 cm interval. Conditioning on the two western archives also lowers the predicted median at this independent core from "+format_cm(select(base,"old-085")["latent_cm"]["median"])+" cm to "+format_cm(select(ref,"old-085")["latent_cm"]["median"])+" cm. This illustrates the risk of shrinking several poorly sampled bearings together. A measured loading could test that particularly thin tail. [Chen et al., Table 1](https://doi.org/10.1029/2021JB023243)","",
           "Two additional Pacific core records consist of spots. The comparable-bed reconstruction gives "+format_cm(select(ref,"old-045")["latent_cm"]["median"])+" cm at YK14-01 PC09 and "+format_cm(select(ref,"old-069")["latent_cm"]["median"])+" cm at NT13-19 PL01. Their patch dimensions cannot calibrate original bulk fallout. These sites and Lake Kushu remain independent probes in all western refits. [Ikehara et al., Tables 1 and 3](https://doi.org/10.1016/j.quaint.2017.08.022)",""]
    direct=output/"spherical-refit/report.json"
    if direct.exists():
        d=read(direct)
        q=d["volume_km3"];w=d["importance_volume_km3"]
        rows+=["## Independent spherical refit","",
               f"A fresh four-chain fit using the comparable-bed scenario gives **{q['median']:.1f} km³** (conditional 95% interval **{q['lower95']:.1f}–{q['upper95']:.1f} km³**). Updating the independent thickness-only posterior gave {w['median']:.1f} km³ ({w['lower95']:.1f}–{w['upper95']:.1f}). The upper-tail estimates differ by {abs(q['upper95']-w['upper95']):.0f} km³. Both numerical results are retained. The direct sampler and angular-normalization assessment "+("passed." if d["diagnostics"]["passed"] else "has a warning.")+" [Independent posterior and diagnostics](spherical-refit/report.json) · [All direct-refit draws](spherical-refit/posterior.npz)",""]
    q3=next((f["archive_only_holdout"] for f in geo["quadrants"]["folds"] if "archive_only_holdout" in f),None)
    if q3 and "joint_predictive_probability" in q3:
        rows+=["## Withhold the entire western archive sector","",
               f"No measured thickness falls in the northwest source quadrant. With both western cores withheld, the measured-fan model assigns probability **{q3['joint_predictive_probability']:.1%}** to both reported no-bed outcomes under the comparable-bed transfer. The test uses {len(q3['quantitative_training_row_ids'])} quantitative training rows and {len(q3['archive_training_ids'])} archive readings. Geographical blocking around the actual archives is recorded. The independent thickness-only spherical family assigns {sphere['visible_analogs']['joint_reading_probability']:.1%} to the same joint observation. [All quantitative and archive holdouts](visible_analogs/validation.json)",""]
    (output/"field-evidence.md").write_text("\n".join(rows))
    p=output/"README.md"
    lead="**[What the Longgang and Quanyang cores reject](field-evidence.md)** compares the original predictions, the conditional refits and the archive observations."
    s=p.read_text().splitlines()
    if lead not in s:s[1:1]=["",lead]
    p.write_text("\n".join(s)+"\n")
    write_json(output/"artifacts.json",{p.relative_to(output).as_posix():digest(p) for p in output.rglob("*") if p.is_file() and p!=output/"artifacts.json"})
