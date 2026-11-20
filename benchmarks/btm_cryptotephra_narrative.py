"""Write the B-Tm comparison from fitted volumes and held-out predictions."""
import json

import pandas as pd


FAMILIES = {"stacked":"GP ensemble", "spline":"Cubic spline", "directional":"Uncertain fan direction",
            "spherical":"Spherical mixture"}
CASES = {"source_only":"Previous, 68 directly quantitative sites","principal":"Principal, site-derived limits",
         "one_tenth_response":"Tenfold weaker limits","recognition_analogs":"Visible-bed analogues, 1 / 2 / 1 cm"}


def read(path):
    return json.loads(path.read_text())


def number(value):
    if pd.isna(value):
        return "outside hull"
    return f"{value:.3g}" if value < 1 else f"{value:,.2f}"


def write_study(home,out,volumes,sites,held):
    regional = read(home/"report.json")
    az = read(home/"directional/azimuth/report.json")
    sphere = read(home/"spherical-mixture/report.json")
    vol = volumes.set_index(["case","method"])
    loc = sites.set_index(["case","method","id"])
    primary = vol.loc["principal","directional"]
    previous = vol.loc["source_only","directional"]
    ids = [s["id"] for s in regional["input"]["interpreted_upper_limits"]]
    lines = ["# How cryptotephra changes the B-Tm reconstruction", "",
             f"The principal directional median map integrates to **{primary.median_map_km3:,.1f} km³** of bulk airborne fallout, compared with **{previous.median_map_km3:,.1f} km³** on the preceding 68-site input. Complete simulated deposits have a median of **{primary.volume_median_km3:,.1f} km³** and a conditional 95% interval of **{primary.lower95:,.1f}–{primary.upper95:,.1f} km³**. The central map integrates the pointwise median thickness. The interval integrates each spatially coherent deposit separately.",
             "", "The three additions are positive occurrences with unknown primary bed thickness. The principal fits use conditional upper limits of 0.06 cm at Lake Sihailongwan, 0.64 cm at Quanyang peat and 0.45 cm at Lake Kushu. The lake limits transfer a same-eruption count response from Suigetsu. The peat limit assumes full retention in a three-slice dry-mass budget, a maximum host dry density and a minimum fallout bulk density. Local variation enters the censoring likelihood. An interpreted bound gives neither an exact thickness nor a numerical lower endpoint. [Primary sources and calculations](../../../../data/audits/B-Tm/README.md) · [Accepted observations](../../../../data/eruptions/B-Tm_cryptotephra.csv)",
             "", "![B-Tm directional median isopachs before and after the cryptotephra evidence](directional-before-after.png)",
             "", "## What changed at the three sites?", "",
             "Each entry gives the previous and revised latent median thickness in cm. A regional fit ends at its own observation hull. The revised values at the three fitted sites show the effect of conditioning. [Full site intervals and alternative responses](four-method-site-predictions.csv)",
             "", "| Method | Sihailongwan | Quanyang | Kushu |","|---|---:|---:|---:|"]
    for method,label in FAMILIES.items():
        changes = []
        for sid in ids:
            before, after = (loc.at[(case,method,sid),"latent_median_cm"] for case in ("source_only","principal"))
            changes.append(f"{number(before)} → {number(after)}")
        lines.append("| "+label+" | "+" | ".join(changes)+" |")
    peat = held[(held.method == "spherical") & (held.site == "Quanyang peat")].iloc[0]
    peat_spline = held[(held.method == "spline") & (held.site == "Quanyang peat")].iloc[0]
    lines += ["", f"The spherical median at Quanyang is {loc.at[('principal','spherical','quanyang'),'latent_median_cm']:.2f} cm. Its fitted probability that a repeat local observation would fall below {peat.limit_cm:.2f} cm is {peat.fitted_repeat_probability_of_bound:.1%}. The spline gives the same repeat reading only {peat_spline.fitted_repeat_probability_of_bound:.1%} probability, even after fitting the core. These limits enter the observation likelihood with local scatter. They can pull an isopach inward while leaving disagreement at the site. [Fitted and withheld probabilities for all three limits](heldout-archive-probabilities.csv)"]
    thresholds = pd.read_csv(out/"response-required-by-contours.csv")
    old_response = thresholds[thresholds.comparison.eq("source_only")].set_index("id")
    lines += ["", f"To reconcile the old Sihailongwan median with the observed glass profile, the lake's effective shard response would have to be at most **{old_response.at['longgang','maximum_compatible_response']:.2%}** of the Suigetsu reference. At Quanyang, the old median permits at most **{old_response.at['quanyang','maximum_compatible_response']:.1%}** retention under the stated mass and density assumptions. The profile sums and mass budget can be used to test any proposed isopach or archive response. [Response required by particular contours](response-required-by-contours.csv)",
              "", "### Withhold the core's geographic region", "",
              "The table gives the probability of a local observation below the stated upper limit when the entire buffered geographic fold containing that core was withheld. These predictions include local thickness scatter. The regions and training memberships match across all four families. The transfer assumptions are fixed before the folds are scored. Because the two lake bounds share a reference at Suigetsu, these folds do not independently validate the count conversion. Each probability tests the specified archive interpretation. It is a one-sided check and can reward a model that predicts extremely thin deposits.",
              "", "| Site | Upper (cm) | Fold | GP | Spline | Directional | Spherical |","|---|---:|---:|---:|---:|---:|---:|"]
    for item in regional["input"]["interpreted_upper_limits"]:
        subset = held[held.row_id.eq(item["row_id"])].set_index("method")
        values = [subset.at[m,"heldout_probability_of_bound"] for m in FAMILIES]
        folds = subset.fold.dropna().unique()
        assert len(folds) <= 1
        shown = [("unscored" if pd.isna(p) else f"{p:.2%}") for p in values]
        lines.append(f"| {item['name']} | {item['limit_cm']:g} | {int(folds[0]) if len(folds) else '—'} | "+" | ".join(shown)+" |")
    western = held[held.site.eq("Lake Sihailongwan")].set_index("method")
    lines += ["", f"With Sihailongwan and Quanyang both withheld in the western fold, the directional local median at Sihailongwan returns to **{western.at['directional','heldout_local_median_cm']:.2f} cm**. The directional model assigns the 0.06 cm upper observation {western.at['directional','heldout_probability_of_bound']:.1%} probability. The westward retreat therefore relies on the cores. The GP and spline give Sihailongwan local medians of {western.at['stacked','heldout_local_median_cm']:.2g} and {western.at['spline','heldout_local_median_cm']:.2g} cm. Values that small can score well against an upper limit. The positive glass occurrence establishes arrival, while its magnitude remains conditional on archive response.",
              "", "[Scores on 64 exact thicknesses and on 67 direct records excluding the Suigetsu reference](heldout-by-evidence.csv) · [Individual archive scores](heldout-archive-probabilities.csv) · [All regional holdouts](../heldout_predictions.csv) · [All directional holdouts](../directional/azimuth/heldout.csv) · [All spherical holdouts](../spherical-mixture/heldout_predictions.csv)",
              "", "## Where the contours move", "",
              "Distances are kilometres along a ray from Paektu to its first outward median-contour crossing. A detached lobe farther along the ray is excluded. The two fits have the same six structures and the same target-excluded calibration of structure and fan direction. The rows below show the western bearings. [All eight bearings and all fitted levels](directional-reach.csv)",
              "", "| Direction | Level (cm) | Previous reach (km) | Revised reach (km) |", "|---|---:|---:|---:|"]
    reach = pd.read_csv(out/"directional-reach.csv").pivot(index=["sector","level_cm"],columns="case",values="first_crossing_great_circle_km")
    for (sector,level),r in reach.iterrows():
        if sector in ("W","NW","SW") and level in (.1,1.,10.):
            def km(value):
                return "unresolved" if pd.isna(value) else f"{value:,.0f}"
            lines.append(f"| {sector} | {level:g} | {km(r.source_only)} | {km(r.principal)} |")
    marine = pd.read_csv(out/"marine-occurrence-check.csv").set_index(["case","method"])
    lines += ["", "### Independent marine occurrence", "",
              "At Ulleung Basin core 13PT-P4_026, the 2 cm sediment slice contains 9,529 shards/g. Applying the same lake reference gives 0.0762 cm for that slice. Bioturbation, grain sorting and lateral transport make this an illustrative local equivalent. There is no verified primary-fallout bound at the core. It lies outside both regional hulls.",
              "", "| Global method | Previous latent median (cm) | Revised latent median (cm) | Previous probability of ≥0.0762 cm | Revised probability |","|---|---:|---:|---:|---:|"]
    for m in ("directional","spherical"):
        med = [number(loc.at[(c,m,"old-085"),"latent_median_cm"]) for c in ("source_only","principal")]
        probs = [marine.at[(c,m),"p_local_at_least_equivalent"] for c in ("source_only","principal")]
        lines.append(f"| {FAMILIES[m]} | {med[0]} | {med[1]} | {probs[0]:.2%} | {probs[1]:.2%} |")
    lines += ["", "These probabilities include site variability. A low exceedance probability signals possible narrowing at sea **if** the reference transfers to this core. The glass occurrence remains an independent check. [Marine calculations](marine-occurrence-check.csv) · [Source profile](https://doi.org/10.1029/2021JB023243)",
              "", "## Bulk volume under alternative archive responses", "",
              "Volumes are km³ of bulk fallout inferred from the reported thicknesses. The study applies no universal decompaction correction. Each cell below gives the median of complete volume draws and their conditional 95% interval, except the separate column for the integrated directional median map. GP and spline cover their own regional hull. The directional ensemble and spherical mixture integrate over the Earth.",
              "", "| Archive assumption | GP, hull | Spline, hull | Directional median map | Directional complete deposits | Spherical complete deposits |",
              "|---|---:|---:|---:|---:|---:|"]
    for case,label in CASES.items():
        cells = []
        for m in FAMILIES:
            r = vol.loc[(case,m)]
            if m == "directional":
                cells.append(f"{r.median_map_km3:,.1f}")
            cells.append(f"{r.volume_median_km3:,.1f} [{r.lower95:,.1f}–{r.upper95:,.1f}]")
        lines.append("| "+label+" | "+" | ".join(cells)+" |")
    regional_cases = pd.read_csv(out/"regional-sensitivities.csv").set_index(["case","method"])
    earlier_gp = regional_cases.loc[("source_only","stacked")]
    current_gp = regional_cases.loc[("principal","stacked")]
    current_spline = regional_cases.loc[("principal","spline")]
    footprint = read(home/"support/report.json")["support"]["standard"]["area_km2"]
    tail = read(out/"spline-tail.json")
    refined = tail["joint_draw_check"][-1]
    q025,q50,q975 = refined["conditional_quantiles_km3"]
    upper_splits = [q[-1] for q in refined["half_sample_quantiles"]]
    lines += ["", "![Conditional volume intervals under the four archive interpretations](volume-sensitivity.png)", "", "The weaker response allows upper limits of 0.60, 6.4 and 4.5 cm. The visible-bed analogue uses 1, 2 and 1 cm at the same sites. Each sensitivity refits the local censored likelihood. The preceding case omits the three inferred limits. All four retain the reported Suigetsu limit. The shared lake response is fixed within each scenario. The volume intervals condition on that choice and no probability weights are assigned to the four scenarios. [All estimands](four-method-sensitivities.csv) · [Source inputs and hashes](../../../../data/audits/B-Tm/cryptotephra-principal.json)",
              "", "## Regional footprint and the spline's upper tail", "",
              f"Adding the cores expands the hull from {earlier_gp.fitted_hull_km2:,.0f} to {current_gp.fitted_hull_km2:,.0f} km². On the original footprint the GP mean changes from {earlier_gp.expected_km3:,.2f} to {current_gp.original_hull_expected_km3:,.2f} km³. Including the new ground gives {current_gp.expected_km3:,.2f} km³. The standard interpolation footprint defined by exact thickness sites covers {footprint:,.0f} km². [Matched-domain calculations](regional-sensitivities.csv) · [Footprint and contour screen](../support/README.md)",
              "", f"The selected cubic spline gives a formal regional mean of {current_spline.expected_km3:.3g} km³. Its log-thickness standard deviation reaches {tail['peak_node_latent_log_sd']:.2f} near the bounded Lake Suigetsu site. The highest-contributing 1% of hull area contributes {tail['share_expected_volume_from_top_one_percent_area']:.5%} of the mean. The fitted surface's median thickness at the peak-expectation quadrature node is {tail['peak_node_median_cm']:.2g} cm. This instability also occurs on the original footprint. Holding the preceding spline tuning fixed on all 71 observations gives an analytical mean of {tail['previous_settings_on_principal_data']['expected_km3']:.1f} km³. Model selection and poorly constrained local variance both matter.",
              "", f"The retuned spline's median-map integral is {current_spline.median_map_km3:.2f} km³. In a check with {refined['draws']:,} complete draws and {len(tail['joint_draw_check'])} integration orders, the finest rule gives a volume median of {q50:.1f} km³ and a conditional 95% interval of {q025:.1f}–{q975:.1f} km³. Its two half-sample upper endpoints are {upper_splits[0]:.0f} and {upper_splits[1]:.0f} km³. The ordinary 2,000-draw run appears in the main table. The GP predicted the withheld regions better on this input. Both regional families remain visible. [Spline numerical checks](spline-tail.json) · [Six-eruption matched regional benchmark](matched-regional.csv) · [Benchmark and synthetic-trial checks](regional-benchmark-check.json)",
              "", "![Regional spline log-thickness uncertainty on both survey footprints](spline-uncertainty.png)",
              "", "## Numerical and geographic limits", ""]
    gp_integral = regional["methods"]["stacked"]["regional_volume"]
    if not gp_integral["converged"]:
        delta = gp_integral["refinement"][-1]["relative_change"]
        lines += [f"The GP hull integral exhausted its standard refinement sequence. The final change was {delta:.3%}. The integral still carries the repository's numerical warning. [All integration orders](../report.json)", ""]
    numerical = az["volume"]["numerical"]
    checks = ["spatial quadrature","split-packet map","split-packet volume quantiles"]
    flags = [numerical[k] for k in ("spatial_passed","split_map_passed","joint_quantile_split_passed")]
    full = sum(s["diagnostics"]["passed"] for s in az["members"].values())
    lines += [f"The full-data directional structures passed sampling checks in {full} of {len(az['members'])} fits. "
              +", ".join(f"{name} {'passed' if flag else 'retained a warning'}" for name,flag in zip(checks,flags,strict=True)).capitalize()+".",
              "", "| Validation or prior family | Warning or unestimated fits |","|---|---|"]
    for kind in ("geographic","quadrants"):
        folds = az["validation"][kind]["folds"]
        flags = [f"{r['fold']} ({r['status']})" if not r["test"] or "revised_score" not in r else f"{r['fold']} (sampling or phase check)"
                 for r in folds if not r["test"] or "revised_score" not in r or not (r.get("passed") and r.get("phase_passed"))]
        lines.append(f"| Directional {kind} holdouts | {', '.join(flags) or 'All passed'} |")
    failing = [str(f["fold"]) for f in sphere["validation"]["fits"] if not f["fit"]["diagnostics"]["passed"]]
    lines.append(f"| Spherical geographic holdouts | {', '.join(failing) or 'All passed'} |")
    warning = [k for k,v in sphere["fits"].items() if not v["diagnostics"]["passed"]]
    lines.append(f"| Spherical full-data assumptions | {', '.join(warning) or 'All passed'} |")
    lo,hi = sphere["sensitivity_envelope_km3"]
    lines += ["",f"Across the eight spherical prior fits, the outer conditional interval endpoints span {lo:,.1f}–{hi:,.1f} km³. This envelope measures sensitivity to the published priors. Numerical warnings are retained. [Prior-by-prior estimates](../spherical-mixture/estimates.csv) · [Diagnostics, held-out fits and normalization](../spherical-mixture/report.json)"]
    screens = read(home/"support/report.json")["methods"]
    screens = [screens[m]["contours"] for m in ("stacked","spline")] + [
        read(home/"directional/azimuth/complete-contour-audit.json"),read(home/"spherical-mixture/complete-contour-audit.json")]
    bands = [(m,k,v["selected_band"]) for m,s in zip(FAMILIES,screens,strict=True) for k,v in s["policies"].items()]
    qualified = [f"{FAMILIES[m]} ({p})" for m,p,band in bands if band is not None]
    lines += ["", f"{len(qualified)} of {len(bands)} family–policy combinations support a complete isopach band."
              + (" Qualifying combinations include "+", ".join(qualified)+"." if qualified else "")
              +" Measured-bracketed fragments have a different evidential status from continuation through gaps. Both global estimates condition on the geometry of unsampled fallout.",
              "", "## Reproduce the comparison", "",
              "From the repository root with the project installed, write each run into a fresh directory. The complete comparison includes all geographic holdouts, fixed prior sensitivities and contour screens. [Model intake and environment](../../../../docs/NEW_ERUPTION.md)",
              "", "~~~sh",
              "python -m benchmarks.btm_cryptotephra --output results/btm-counts",
              "python -m benchmarks.btm_cryptotephra_evidence --output results/btm-counts",
              "tephra-isopachs compare B-Tm --output results/btm-principal",
              "for c in one_tenth_response recognition_analogs; do",
              '  tephra-isopachs compare "data/audits/B-Tm/cryptotephra-$c.csv" --regional-only --output "results/btm-$c"',
              "  for family in directional spherical; do",
              '    python -m benchmarks.btm_cryptotephra_global --case "$c" --family "$family" --work "results/btm-fits/$c/$family" --output results/btm-core-study',
              "  done","done",
              "python -m benchmarks.btm_cryptotephra_regional --previous archive/2026-11-19-b-tm-thickness-only/results --primary results/btm-principal --weaker results/btm-one_tenth_response --analog results/btm-recognition_analogs --output results/btm-core-study",
              "python -m benchmarks.regional --output results/btm-matched-benchmark",
              "python -m benchmarks.btm_cryptotephra_report --home results/btm-principal --matched-benchmark results/btm-matched-benchmark --output results/btm-core-study",
              "python -m benchmarks.verify_btm_cryptotephra","~~~","",
              "[Case maps and summary](../../README.md) · [Previous 68-site reconstruction](../../../../archive/2026-11-19-b-tm-thickness-only/README.md) · [File identities](release.json)", ""]
    (out/"README.md").write_text("\n".join(lines))
