"""Readable comparisons of complete-deposit intervals, validation and extrapolation."""
import numpy as np
from src.empirical_isopach.models.nested.geometry import DIAMETER


def write_report(folder, rows, experiments, reports, challenges, areas, recovery):
    families = ("Directional", "Spherical", "Nested")
    out = ["# Whole-deposit volume from Bayesian nested isopachs", "",
           "Each row compares the same accepted thickness measurements and bounds. Each interval integrates complete simulated deposits over the Earth. The directional case home also reports the integral of its median map. Those two summaries can differ appreciably when a fan can turn.", "",
           "![Whole-deposit median volumes and conditional intervals](volumes.png)", "",
           "| Eruption | Directional median | Spherical median | Nested median [conditional 95%], km³ | Nested site coverage | Full fit · geographic fits passing |",
           "|---|---:|---:|---:|---:|---|"]
    for slug, report in reports.items():
        group = [r for r in rows if r["slug"] == slug]
        lo, med, hi = report["conditional_volume_km3"]
        cov = report["exact_site_coverage95"]
        status = "passed" if report["population_check_passed"] else "**warning**"
        path = f"../../eruptions/{slug}/results/nested-isopachs/README.md"
        out.append(f"| [{report['name']}]({path}) | {group[0]['median_km3']:,.1f} | {group[1]['median_km3']:,.1f} | "
                   f"**{med:,.1f}** [{lo:,.1f}–{hi:,.1f}] | {cov['covered']}/{cov['exact_sites']} | {status} · {report['geographic_folds_passed']}/4 |")
    scores = [np.mean([r["equal_region_log_score"] for r in rows if r["method"] == f]) for f in families]
    bands = sum(bool(x["supported_band_policies"]) for x in reports.values())
    out += ["", f"The equal-eruption mean geographic negative log scores are {scores[0]:.3f} for directional, {scores[1]:.3f} for spherical and {scores[2]:.3f} for nested. Each case gives its four buffered regions equal weight. Lower scores indicate better prediction at the withheld sites. Complete contour bands qualified under at least one nested support policy in {bands} of {len(reports)} cases. Ancient global volumes have no independent measured total here. [Case volumes and scores](cases.csv) · [Matched regional rerun](regional-replay.csv)", "",
            "## Where the nested estimate comes from", "",
            "| Eruption | Median volume fraction in measured triangles | Median fraction beyond the farthest exact site | That distal fraction among the largest 5% of volumes | Median if exponential / power tail (km³) |",
            "|---|---:|---:|---:|---:|"]
    for d in reports.values():
        x = d["spatial_volume_budget"]
        tail = d["volume_conditional_on_tail"]
        med = [f"{tail[k][1]:,.0f}" if tail[k] is not None else "—" for k in ("exponential","power")]
        out.append(f"| {d['name']} | {x['fraction_by_part'][0][1]:.1%} | {x['fraction_by_part'][2][1]:.1%} | "
                   f"{x['highest_5pct_by_total_median_fraction'][2]:.1%} | {' / '.join(med)} |")
    out += ["", "The circle through the farthest exact measurement is a diagnostic. Individual parts of that circle can have very different sampling density. Each case reports the volume per compass sector, the full profile and archive checks. Minoan includes the accepted Plinian and co-ignimbrite fans together.","",
            "## What changed in the nested fit","",
            "Three versions use identical observations and geographic training partitions. The initial version put a prior on thinning slopes. The second sets a prior on the ratio of distal to core volume using independently mapped deposits. The third uses that fitted profile throughout the observed area ranks and transfers a separate remaining mass budget beyond the last sampled rank. The last two produce exactly the same fitted thickness at every training site. The test in the last column withholds the farthest fifth of observations by source distance and removes nearby training sites with the standard geographic buffer. Lower scores indicate better prediction.","",
            "| Eruption | Initial median (km³) | Distal/core prior alone | Refined median | Fraction beyond last sampled rank, refined | Farthest-fifth scores, initial / prior alone / refined |",
            "|---|---:|---:|---:|---:|---|"]
    for slug,d in reports.items():
        qa=d["initial_profile"]["conditional_volume_km3"][1]
        t=d["terminal_continuation"]
        scores = [d["outer_fifth"]["fits"][key]["equal_site_negative_log_score"] for key in ("initial","distal_budget","continued")]
        out.append(f"| {d['name']} | {qa:,.1f} | {t['unrestricted_volume_km3'][1]:,.1f} | {d['conditional_volume_km3'][1]:,.1f} | "
                   f"{t['volume_fraction_beyond_last_sampled_rank'][1]:.1%} | {' / '.join(f'{x:.3f}' for x in scores)} |")
    out += ["","### Uncertainty across three thinning assumptions", "",
            "The span below goes from the smallest lower endpoint to the largest upper endpoint across the three nested fits. Each individual interval is conditional on its own transport assumptions. The combined span has no assigned posterior probability. It shows how much narrower an answer can appear when only one thinning law is reported.", "",
            "| Eruption | Span of conditional 95% endpoints (km³) | Range of scenario medians (km³) |",
            "|---|---:|---:|"]
    for d in reports.values():
        variants = np.array([d["initial_profile"]["conditional_volume_km3"],
                             d["terminal_continuation"]["unrestricted_volume_km3"],d["conditional_volume_km3"]])
        out.append(f"| {d['name']} | {min(variants[:,0]):,.1f}–{max(variants[:,2]):,.1f} | "
                   f"{min(variants[:,1]):,.1f}–{max(variants[:,1]):,.1f} |")
    out += ["","The initial Aso-4 and Ilopango posteriors placed a median 81% and 96% of their volumes beyond the farthest exact site. Aso-4's median rose with the distal/core prior alone. Applying the remaining-mass transfer exposes the volume that could still lie inside broadly continued, poorly sampled directions. The four mapped donor volcanoes offer limited evidence for giant co-ignimbrite falls. Their distal thickness profiles are reconstructions with their own extrapolation error. [Archived initial estimates and draws](../../archive/2026-11-21-nested-initial/README.md) · [Equations and donor assumptions](../../docs/methods/nested-isopachs.md)","",
            "## Experiments with the two existing methods","",
            "| Family | Mean control score | Mean candidate score | Eruptions improved / tied / worsened | All numerical checks passed |",
            "|---|---:|---:|---:|---:|"]
    aso = reports["aso-4"]
    wings = [r for r in aso["sectors"] if r["sector"] in ("SE","S","W")]
    examples = ", ".join(f"{r['sector']} {r['median_volume_fraction']:.1%} ({r['exact_sites']} exact)" for r in wings)
    sampled_band = np.pi*DIAMETER**2
    extent = aso["area_km2_at_cm"]["0.1"][1]/sampled_band
    where = out.index("## Experiments with the two existing methods")
    out[where:where] = [f"At Aso-4 the refined model assigns the following median shares of complete-deposit volume to poorly sampled sectors. The shares are {examples}. Its median area above 1 mm covers {extent:.1%} of the Earth. These values expose the direction and scale of the remaining extrapolation. [Compass-sector evidence for all eruptions](sector-evidence.csv)",""]
    if all("edge_deletion" in d for d in reports.values()):
        group = ["## Influence of the site setting the last sampled isopach", "",
                 "The test identifies the row setting the outermost sampled rank most often across posterior draws. It refits after withholding that row, then after withholding its local buffered cluster. A change reflects the observations and the revised geographic rank at which the analogue mass budget begins. These diagnostic deletions preserve the original accepted dataset.", "",
                 "| Eruption | Principal median (km³) | One site withheld | Local cluster withheld | Cluster size | Single / cluster numerical checks |",
                 "|---|---:|---:|---:|---:|---|"]
        for slug,d in reports.items():
            trials = d["edge_deletion"]["fits"]
            one,cluster = trials["single"],trials["cluster"]
            status = ["passed" if trial["numerical"]["passed"] else "warning" for trial in (one,cluster)]
            path=f"../../eruptions/{slug}/results/nested-isopachs/edge-deletion/README.md"
            group.append(f"| [{d['name']}]({path}) | {d['conditional_volume_km3'][1]:,.1f} | "
                         f"{one['volume_km3'][1]:,.1f} | {cluster['volume_km3'][1]:,.1f} | {len(cluster['test_rows'])} | {' / '.join(status)} |")
        out[where:where] = [*group,""]
    pos = out.index("## Experiments with the two existing methods")
    distal_lines = ["## Testing extra thinning beyond the observations","",
                    "A further sensitivity makes ash thin progressively faster beyond the largest equal-area rank of any training site. The three scenarios have identical fitted likelihoods and each integrates over the whole Earth. Stronger or weaker distal removal can change total volume without leaving a measurable signal at the fitted sites.","",
                    "| Eruption | Original median | One extra sampled radius | Two | Four | Geographic score, original → one-radius |",
                    "|---|---:|---:|---:|---:|---:|"]
    for d in reports.values():
        t = d["distal_rolloff"]
        q = [t[k]["volume_km3"][1] for k in ("1.0","2.0","4.0")]
        distal_lines.append(f"| {d['name']} | {d['conditional_volume_km3'][1]:,.1f} | {q[0]:,.1f} | {q[1]:,.1f} | {q[2]:,.1f} | "
                            f"{d['geographic_score']:.3f} → {t['1.0']['equal_region_negative_log_score']:.3f} |")
    bound = next(b for b in reports["b-tm"]["fitted_bound_checks"] if b["name"]=="Lake Sihailongwan")
    distal_lines += ["",f"Lake Sihailongwan has a conditional upper bound of {bound['upper_cm']:.2f} cm in the accepted B-Tm table. The nested median field gives {bound['latent_cm'][1]:.2f} cm there, and the fitted model assigns the upper-bound event a probability of {bound['predictive_probability_of_event']:.2%}. Its western contours merit particular caution. [B-Tm bounds and predictions](../../eruptions/b-tm/results/nested-isopachs/README.md)",""]
    distal_lines.extend(["## How sensitive are the nested volumes to unmeasured bearings?", "",
                         "This comparison keeps the fitted thickness at every training site. It allows a hidden fan to turn in poorly sampled bearings. A separate high-ash scenario increases its thickness along its unknown axis. The two columns give the medians of the corresponding complete deposits.", "",
                         "| Eruption | Principal (km³) | Uncertain bearing (km³) | Enhanced hidden fan (km³) | Paired site scores, principal / uncertain bearing |",
                         "|---|---:|---:|---:|---:|"])
    for slug,d in reports.items():
        study=d["bearing_study"]
        score=study["paired_scores"]
        link=f"../../eruptions/{slug}/results/nested-isopachs/bearing-study/README.md"
        distal_lines.append(f"| [{d['name']}]({link}) | {study['baseline_km3'][1]:,.1f} | "
                            f"{study['retained_volume_km3'][1]:,.1f} | {study['enhanced_hidden_fan_km3'][1]:,.1f} | "
                            f"{score['reference_score']:.3f} / {score['bearing_score']:.3f} |")
    distal_lines.append("")
    out[pos:pos] = distal_lines
    for f in families[:2]:
        sub = [r for r in experiments if r["method"] == f]
        delta = np.array([r["candidate_score"]-r["control_score"] for r in sub])
        out.append(f"| {f} | {np.mean([r['control_score'] for r in sub]):.3f} | {np.mean([r['candidate_score'] for r in sub]):.3f} | "
                   f"{sum(delta < -1e-9)} / {sum(abs(delta) <= 1e-9)} / {sum(delta > 1e-9)} | {sum(r['numerical_checks_passed'] for r in sub)}/{len(sub)} |")
    out += ["", "For directional deposits the new sensitivity allows the amount of ash in a poorly sampled bearing to vary coherently across a draw. For the spherical mixture it gives less prior weight to fans whose centres are far from the vent relative to their widths. B-Tm’s source-focused spherical geometry receives exactly unit weight. Each paired control uses the same sample of posterior fields as its candidate. Both changes are published as named sensitivities. The changes in predictive score and remaining parent-fit warnings give insufficient grounds to promote either as the new principal estimate. [Per-case changes and conditional intervals](peer-sensitivities.csv) · [Equations and priors](../../docs/methods/continuation-sensitivities.md)", "",
            "## Known-volume deposits", "",
            "The original suite has six deposit and sampling designs. They include a missing proximal core, an unsampled sector, two fans and a diffuse distal component. The second suite also uses identical sample values with two possible true deposits, one carrying a lobe invisible to every sample. Each seed and sampling design counts once. Its two possible truths each count half in the averages. Predictions are conditional nominal 95% intervals. Coverage shows the unweighted count of displayed truths. Mean absolute log error measures central-volume error. The interval score adds log interval width to 40 times the log distance of a truth outside its interval. Smaller values are better. Numerical warnings remain in all comparisons.", "",
            "| Sampling seeds | Family | Truths covered | Mean absolute log error | Mean log interval width | Mean log interval score | Fits with warnings / designs |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for seeds in ((10,11),(90,91)):
        for f in (*families, "Nested initial","Directional gap amount", "Spherical source prior"):
            part = [r for r in challenges if r["seed"] in seeds and r["method"] == f]
            if not part:
                continue
            def avg(k):
                return np.average([r[k] for r in part],weights=[r["weight"] for r in part])
            designs = {(r["scenario"],r["seed"]):r["numerical_checks_passed"] for r in part}
            out.append(f"| {seeds[0]}, {seeds[1]} | {f} | {sum(r['covered'] for r in part)}/{len(part)} | "
                       f"{avg('absolute_log_error'):.3f} | {avg('log_interval_width'):.3f} | {avg('log_interval_score'):.3f} | "
                       f"{sum(not v for v in designs.values())}/{len(designs)} |")
    diffuse = [r for r in challenges if r["scenario"]=="diffuse_global_tail" and r["method"]=="Nested"]
    if diffuse:
        medians = [r["median_km3"] for r in diffuse]
        truth = diffuse[0]["truth_km3"]
        out += ["", f"The simulated diffuse distal deposit contained {truth:g} km³. Across four independent sampling seeds the refined nested medians were {min(medians):.1f}–{max(medians):.1f} km³. Its conditional 95% interval covered the truth in {sum(r['covered'] for r in diffuse)}/{len(diffuse)} cases. Historical thinning donors provide weak protection against a thin, widespread component. A revised central estimate for Aso-4 or Ilopango cannot establish that this failure mode is absent."]
    angles = [r for r in challenges if r["scenario"]=="invisible_sector" and r["seed"]==90 and r["method"]=="Nested"]
    if len(angles)==2:
        truths = sorted(r["truth_km3"] for r in angles)
        a=angles[0]
        out += ["", f"For one identical set of thickness readings, two deposits contain {truths[0]:.1f} and {truths[1]:.1f} km³. The nested fit necessarily gives both the same median of {a['median_km3']:.1f} km³ and the same interval of {a['lower95_km3']:.1f}–{a['upper95_km3']:.1f} km³. This tests the ability of an interval to accommodate an unobserved lobe."]
    out += ["", "Contour recovery is evaluated independently at 1, 10 and 100 cm. The table shows the mean ratio of area correctly predicted above threshold to area above threshold in either the prediction or truth. It includes only paired cases with at least 64 effective quadrature nodes in both unions. The two hidden-lobe truths share the weight of one sampling design. The detailed table records overextended and missed area separately. Pairings use the published geographic quadrature for each suite.", "",
            "| Seeds | Comparator | Level (cm) | Paired truths | Nested overlap | Comparator overlap |", "|---|---|---:|---:|---:|---:|"]
    for seeds, fam in (((10,11),"Spherical"),((90,91),"Directional")):
        lookup = {(r["scenario"],r["seed"],r["truth"],r["level_cm"],r["method"]):r for r in areas}
        for level in (1.,10.,100.):
            both = [(r,lookup.get((r["scenario"],r["seed"],r["truth"],level,fam))) for r in areas
                    if r["method"]=="Nested" and r["seed"] in seeds and r["level_cm"]==level]
            both = [(u,v) for u,v in both if v and min(u["effective_union_nodes"],v["effective_union_nodes"])>=64 and u["iou"] is not None and v["iou"] is not None]
            if both:
                weights = [.5 if u["scenario"]=="invisible_sector" else 1. for u,v in both]
                a,b = [np.average([pair[i]["iou"] for pair in both],weights=weights) for i in (0,1)]
                out.append(f"| {seeds[0]}, {seeds[1]} | {fam} | {level:g} | {len(both)} | {a:.3f} | {b:.3f} |")
    out += ["", "These designed stress tests reuse published sampling geometries. They exercise failure modes and give no population-level frequency for those modes in nature. The directional, spherical and nested estimates remain separate. A wide conditional interval can cover a true volume even when its median misses a large lobe. The geographically withheld tests and the known-volume tests answer different questions.", "",
            "[Known-volume predictions and truths](known-volumes.csv) · [False and missed isopach areas](known-areas.csv) · [Raw challenge diagnostics](challenge-receipts.json.gz) · [Nested model and prior calibration](../../docs/methods/nested-isopachs.md)"]
    out += ["","## Recovery on the Aso-4 and Ilopango survey layouts","",
            "These four tests were specified after freezing the final model. Only the actual site coordinates and source locations are used. Every thickness and detection limit is simulated. Each design has the same three-component deposit in units of its median survey distance. The paired version adds a localized positive signal to the observed archive while the true primary fallout volume stays fixed. This tests how easily the method mistakes local thickening for primary ash elsewhere.","",
            "| Survey | Local archive thickening | True primary (km³) | Initial median [95%] | Refined median [95%] | Initial / refined checks |","|---|---|---:|---:|---:|---|"]
    for run in recovery:
        first,second=[run["fits"][k] for k in ("initial","continued")]
        def interval(result):
            lo,med,hi=result["volume_km3"]
            return f"{med:,.1f} [{lo:,.1f}–{hi:,.1f}]"
        states=["passed" if x["numerics"]["passed"] else "warning" for x in (first,second)]
        out.append(f"| {run['layout']} | {'yes' if run['local_archive_signal'] else 'no'} | {run['truth']['bulk_km3']:,.1f} | {interval(first)} | {interval(second)} | {' / '.join(states)} |")
    out += ["","The synthetic archive signal is a diagnostic scenario. It supplies no evidence that any particular site was redistributed. The earlier panels use all matched trials, including poorly covered and numerically uncertain outcomes. [Simulation field, observation process and reference integrals](../../benchmarks/nested_deposit_recovery.py) · [Regional known-deposit replay](regional-known.csv)"]
    preview = folder/"pinatubo-preview"
    if preview.is_dir():
        out += ["", "The unregistered 1991 Pinatubo Layer C snapshot has a separate report. [Pinatubo data and estimates](pinatubo-preview/README.md)"]
    (folder/"README.md").write_text("\n".join(out)+"\n")
