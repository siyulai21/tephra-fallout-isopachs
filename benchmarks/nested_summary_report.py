"""Readable comparisons of complete-deposit intervals, validation and extrapolation."""
import numpy as np
from src.empirical_isopach.models.nested.geometry import DIAMETER


def write_report(folder, rows, experiments, reports, challenges, areas, recovery):
    families = ("Directional", "Spherical", "Nested")
    out = ["# Whole-area sensitivity for Bayesian nested isopachs", "",
           "This study uses the broad whole-area continuation throughout. [Principal corridor estimates and maps](README.md)", "",
           "Each row compares the same accepted thickness measurements and bounds. All three point estimates integrate median thickness over Earth. Separate conditional intervals use the 2.5th and 97.5th percentiles of the volume draws.", "",
           "![Median-thickness integrals and conditional intervals](whole-area-volumes.png)", "",
           "| Eruption | Directional map | Spherical map | Nested map [conditional 95%], km³ | Nested site coverage | Full fit · geographic fits passing |",
           "|---|---:|---:|---:|---:|---|"]
    for slug, report in reports.items():
        group = [r for r in rows if r["slug"] == slug]
        lo, hi = report["conditional_volume_interval95_km3"]
        cov = report["exact_site_coverage95"]
        status = "passed" if report["population_check_passed"] else "**warning**"
        path = f"../../eruptions/{slug}/results/nested-whole-area/README.md"
        out.append(f"| [{report['name']}]({path}) | {group[0]['median_map_km3']:,.1f} | {group[1]['median_map_km3']:,.1f} | "
                   f"**{group[2]['median_map_km3']:,.1f}** [{lo:,.1f}–{hi:,.1f}] | {cov['covered']}/{cov['exact_sites']} | {status} · {report['geographic_folds_passed']}/4 |")
    scores = [np.mean([r["equal_region_log_score"] for r in rows if r["method"] == f]) for f in families]
    bands = sum(bool(x["supported_band_policies"]) for x in reports.values())
    out += ["", f"The equal-eruption mean geographic negative log scores are {scores[0]:.3f} for directional, {scores[1]:.3f} for spherical and {scores[2]:.3f} for nested. Each case gives its four buffered regions equal weight. Lower scores indicate better prediction at the withheld sites. Complete contour bands qualified under at least one nested support policy in {bands} of {len(reports)} cases. Ancient global volumes have no independent measured total here. [Case volumes and scores](whole-area-cases.csv) · [Matched regional rerun](regional-replay.csv)", "",
            "## Where the nested estimate comes from", "",
            "| Eruption | Median volume fraction in measured triangles | Median fraction beyond the farthest exact site | That distal fraction among the largest 5% of volumes | 95% interval under exponential / power tails (km³) |",
            "|---|---:|---:|---:|---:|"]
    for d in reports.values():
        x = d["spatial_volume_budget"]
        tail = d["volume_intervals95_by_tail_km3"]
        med = [f"{tail[k][0]:,.0f}–{tail[k][-1]:,.0f}" if tail[k] is not None else "—" for k in ("exponential","power")]
        out.append(f"| {d['name']} | {x['fraction_by_part'][0][1]:.1%} | {x['fraction_by_part'][2][1]:.1%} | "
                   f"{x['highest_5pct_by_total_median_fraction'][2]:.1%} | {' / '.join(med)} |")
    out += ["", "Each eruption’s sensitivity report retains its prior changes, held-out predictions, areas and conditional 95% volume intervals. [Geographic reach and areas](../nested-reach/README.md) · [Sensitivity intervals for the two global peers](peer-sensitivities.csv)", "",
            "## Known-deposit validation", "",
            "| Seeds | Family | Known volumes inside 95% interval | Log interval width | Log interval score | Fits passing |",
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
                       f"{avg('log_interval_width'):.3f} | {avg('log_interval_score'):.3f} | "
                       f"{sum(bool(v) for v in designs.values())}/{len(designs)} |")
    diffuse = [r for r in challenges if r["scenario"]=="diffuse_global_tail" and r["method"]=="Nested"]
    if diffuse:
        truth = diffuse[0]["truth_km3"]
        out += ["", f"The simulated diffuse distal deposit contained {truth:g} km³. Its conditional 95% interval covered the truth in {sum(r['covered'] for r in diffuse)}/{len(diffuse)} cases. Historical thinning donors provide weak protection against a thin, widespread component. A revised central estimate for Aso-4 or Ilopango cannot establish that this failure mode is absent."]
    angles = [r for r in challenges if r["scenario"]=="invisible_sector" and r["seed"]==90 and r["method"]=="Nested"]
    if len(angles)==2:
        truths = sorted(r["truth_km3"] for r in angles)
        a=angles[0]
        out += ["", f"For one identical set of thickness readings, two deposits contain {truths[0]:.1f} and {truths[1]:.1f} km³. The nested fit necessarily gives both the same interval of {a['lower95_km3']:.1f}–{a['upper95_km3']:.1f} km³. This tests the ability of an interval to accommodate an unobserved lobe."]
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
            "| Survey | Local archive thickening | True primary (km³) | Initial 95% interval | Refined 95% interval | Initial / refined checks |","|---|---|---:|---:|---:|---|"]
    for run in recovery:
        first,second=[run["fits"][k] for k in ("initial","continued")]
        def interval(result):
            lo,med,hi=result["volume_km3"]
            return f"{lo:,.1f}–{hi:,.1f}"
        states=["passed" if x["numerics"]["passed"] else "warning" for x in (first,second)]
        out.append(f"| {run['layout']} | {'yes' if run['local_archive_signal'] else 'no'} | {run['truth']['bulk_km3']:,.1f} | {interval(first)} | {interval(second)} | {' / '.join(states)} |")
    out += ["","The synthetic archive signal is a diagnostic scenario. It supplies no evidence that any particular site was redistributed. The earlier panels use all matched trials, including poorly covered and numerically uncertain outcomes. [Simulation field, observation process and reference integrals](../../benchmarks/nested_deposit_recovery.py) · [Regional known-deposit replay](regional-known.csv)"]
    preview = folder/"pinatubo-preview"
    if preview.is_dir():
        out += ["", "The unregistered 1991 Pinatubo Layer C snapshot has a separate report. [Pinatubo data and estimates](pinatubo-preview/README.md)"]
    (folder/"whole-area-study.md").write_text("\n".join(out)+"\n")
