"""Human-readable reports and GIS exports, independent of fitting decisions."""
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contours import build_contours, geographic_contours
from .figures import save_isopach_map, save_diagnostics
from .volume import volume_analysis


CSS = """:root{--ink:#183843;--muted:#5c7075;--teal:#087e8b;--line:#d9e3df;--paper:#fafbf8;--amber:#936025}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1240px;margin:auto;padding:48px 36px 70px}a{color:var(--teal);text-underline-offset:4px}a:hover{color:#154852}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;color:var(--teal)}h1{font-size:clamp(38px,6vw,66px);line-height:1.08;letter-spacing:-.045em;margin:18px 0 20px}h2{font-size:24px;letter-spacing:-.02em;margin:0 0 14px}h3{font-size:18px;margin:0 0 9px}p{margin:0 0 15px}.lead{font-size:19px;color:var(--muted);max-width:800px}.muted,small{color:var(--muted)}small{font-size:13px}
.rule{border-top:1px solid var(--line);margin:32px 0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:32px 0}.stat{padding:22px 24px;background:#edf3ee;border-radius:10px}.stat strong{font-size:32px;font-weight:600;display:block;line-height:1.3;margin:6px 0}.label{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.tag{display:inline-block;border:1px solid #bfd2cc;padding:3px 10px;border-radius:99px;font-size:12px;color:var(--teal);margin-right:5px}.tag.warn{border-color:#dcc7a4;color:var(--amber);background:#fbf6eb}
.panel{border:1px solid var(--line);background:white;border-radius:12px;overflow:hidden;margin:24px 0}.panel img{display:block;width:100%;height:auto}.panel .caption{padding:18px 24px;border-top:1px solid var(--line);font-size:14px;color:var(--muted)}.downloads{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.downloads a{padding:8px 14px;border:1px solid var(--line);border-radius:7px;text-decoration:none;font-size:14px;background:white}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:24px}.card{border:1px solid var(--line);border-radius:12px;background:white;overflow:hidden}.card img{width:100%;display:block}.card .body{padding:22px}.card a{text-decoration:none}.card h2{font-size:28px}.note{border-left:3px solid var(--teal);padding:15px 20px;background:#edf3ee;border-radius:0 7px 7px 0;margin:24px 0;font-size:14px}.note.warn{border-color:#bc863e;background:#faf4e8}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px;background:white;text-align:left}th,td{padding:12px 16px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf3ee;font-size:12px;letter-spacing:.03em}tr:last-child td{border-bottom:0}.number{font-variant-numeric:tabular-nums;white-space:nowrap}details{border-top:1px solid var(--line);padding:18px 0}summary{cursor:pointer;font-weight:600}details p,details ul{margin-top:14px}li{margin:7px 0}code{background:#edf3ee;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}footer{border-top:1px solid var(--line);margin-top:42px;padding-top:20px;font-size:12px;color:var(--muted)}
@media(max-width:720px){main{padding:28px 18px}.grid,.stats{grid-template-columns:1fr}.stats{gap:10px}.stat{padding:16px 20px}.stat strong{font-size:28px}h1{font-size:43px}.lead{font-size:17px}}
@media print{main{padding:0}.downloads,details{display:none}.panel,.card,.stat{break-inside:avoid}body{font-size:12px}h1{font-size:32px}}
"""


def page(title, body):
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)} · Tephra atlas</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>\n")


def number(value):
    return f"{value:,.1f}" if value >= 10 else f"{value:,.2f}"


def method_description(report):
    if "stacking" in report["validation"]:
        return "Spatial holdouts fitted an experimental mixture of exponential and power-law thinning candidates. Mixture quantiles and expected thickness are computed directly; whole-field draws preserve both spatial covariance and model disagreement."
    return f"Spatial validation selected the {report['model']['candidate']['tail']} thinning family within the directional-trend-plus-spatial-residual model."


def empirical_section(report):
    stacking = report["validation"].get("stacking")
    if stacking is None:
        return ""
    rows = "".join(f"<tr><td>{r['candidate']['tail']}</td><td>{r['candidate']['length_factor']:g}</td><td>{r['candidate']['noise_ratio']:g}</td><td>{r['weight']:.1%}</td><td>{r['block_bootstrap_weight_5_95'][0]:.1%}–{r['block_bootstrap_weight_5_95'][1]:.1%}</td></tr>"
                   for r in stacking["components"])
    comparison = "<p>Independent geographic validation was not run. Weight-training scores do not estimate predictive accuracy.</p>"
    nested = report["nested_validation"]
    if nested:
        performance = "".join(f"<tr><td>{label}</td><td>{v['coverage95_observations']:.1%}</td><td>{v['mean_interval_width_log']:.2f}</td><td>{v['mean_log_score']:.3f}</td></tr>"
                              for label, v in (("Ensemble (reported)", nested), ("Extra variance (diagnostic)", nested["discrepancy_adjusted"])))
        comparison = '<div class="table-wrap"><table><thead><tr><th>Outer geographic holdouts</th><th>95% coverage</th><th>Mean interval width · ln cm</th><th>Negative log score · lower is better</th></tr></thead><tbody>'+performance+'</tbody></table></div>'
    return f'''<h2>What did the data support?</h2><p>{method_description(report)}</p>
<div class="table-wrap"><table><thead><tr><th>Tail family</th><th>Length factor</th><th>Noise ratio</th><th>Predictive weight</th><th>Fold reweighting · 5–95%</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="muted">Weights describe predictive contribution, not probabilities that a model is true. Reweighting whole geographic folds tests stability; these ranges are not confidence intervals. Zero-weight candidates remain visible.</p>
{comparison}<div class="note">The final all-data fit learned an extra observation-discrepancy SD of {stacking['calibration']['sd_log']:.3f} in natural-log thickness from its training holdouts. Each outer validation fold estimates its own adjustment independently. The adjustment is exported as a diagnostic: it does not change the headline observation intervals, latent surface or volume draws. Wider intervals alone do not establish better uncertainty. All fits and weights remain estimated, not integrated out.</div>'''


def write_short_report(output, name, report):
    volume = report["volume_analysis"]
    regional = volume["regional"]
    extended = volume["domain_sensitivity"][-1]["selected"]
    nested = report["nested_validation"]
    closed = [r["level_cm"] for r in report["contours"]["levels"] if r["status"] == "closed"]
    omitted = [r["level_cm"] for r in report["contours"]["levels"] if r["status"] == "extends_beyond_map"]
    coverage = (f"Nested geographic holdouts covered {nested['coverage95_observations']:.0%} of observations with nominal 95% intervals; "
                f"natural-log RMSE was {nested['rmse_log']:.2f}." if nested else
                "Independent nested validation was not run; selection scores are not an independent accuracy estimate.")
    bounds = regional['bounds_km']
    region_description = (f"The stated equal-area rectangle is easting {bounds[0]:g} to {bounds[2]:g} km and "
                          f"northing {bounds[1]:g} to {bounds[3]:g} km from the supplied source reference.")
    audit = report['input'].get('audit')
    provenance_note = ("Input selection preserves the supplied audit: "+audit['selection']+
                       " Geological qualifications remain in the source audit; this model run does not re-audit the original publications."
                       if audit else "Input provenance and preservation were not independently audited.")
    predictive_summary = ""
    if nested:
        predictive_summary = ("\n| Geographic holdout metric | Value |\n|---|---:|\n"
                              f"| Negative log score (lower is better) | {nested['mean_log_score']:.3f} |\n"
                              f"| Equal-region negative log score | {nested['equal_region_log_score']:.3f} |\n"
                              f"| Mean 95% interval width (natural-log cm) | {nested['mean_interval_width_log']:.3f} |\n")
    if 'components' in report['model']:
        predictive_summary += "\n| Retained component | Length factor | Noise ratio | Weight |\n|---|---:|---:|---:|\n"
        for component in report['model']['components']:
            candidate = component['model']['candidate']
            predictive_summary += (f"| {candidate['tail']} | {candidate['length_factor']:g} | "
                                   f"{candidate['noise_ratio']:g} | {component['weight']:.1%} |\n")
        predictive_summary += "\nWeights are predictive contributions, not probabilities that models are true. Extra observation-discrepancy adjustment is diagnostic only; see the numerical report.\n"
    text = f"""# {name}: deposit reconstruction

The analysis used **{report['input']['n_measurements']} thickness measurements** and {report['input']['n_upper_limits']} documented upper limits, excluding {len(report['input']['ignored_pseudo_rows'])} legacy pseudo rows. {method_description(report)}

| Result | Estimate |
|---|---:|
| Expected bulk volume in the stated region | {number(regional['mean_km3'])} km³ |
| Integral of the median thickness surface | {number(regional['median_surface_km3'])} km³ |
| Expected volume outside the observation hull | {regional['fraction_expected_volume_outside_data_hull']:.0%} |
| Expected volume after doubling both region widths | {number(extended['mean_km3'])} km³ |
| Last numerical refinement change | {regional['relative_change']:.2%} |

Closed median isopachs were resolved at {', '.join(f'{v:g}' for v in closed)} cm. {'Levels '+', '.join(f'{v:g}' for v in omitted)+' cm remain beyond the map limit and were not artificially closed.' if omitted else 'Extrapolated contour sections are dashed on the map.'}

{coverage}
{predictive_summary}

These are finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. {region_description} {'Numerical convergence passed the 0.5% criterion.' if regional['converged'] else 'Numerical convergence did not meet the 0.5% criterion; this estimate needs refinement.'} Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: {report['model']['uncertainty'].rstrip('.')}. Fitted-parameter uncertainty is not integrated. {provenance_note}

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
"""
    (Path(output)/"short_report.md").write_text(text)
    figures = "\n![Median isopachs with observations and extrapolation](isopachs.png)\n\n![Geographic holdouts and numerical diagnostics](diagnostics.png)\n"
    (Path(output)/"README.md").write_text(text+figures)


def write_dataset_page(output, name, report):
    va = report["volume_analysis"]
    regional = va["regional"]
    nested = report["nested_validation"]
    hull = regional["fraction_expected_volume_outside_data_hull"]
    rows = "".join(f"<tr><td>{r['domain_factor']:g}× widths</td><td class='number'>{number(r['selected']['mean_km3'])}</td>"
                   f"<td class='number'>{number(r['alternative']['mean_km3']) if r['alternative'] else '—'}</td>"
                   f"<td>{'Passed' if r['selected']['converged'] and (r['alternative'] is None or r['alternative']['converged']) else 'Needs refinement'}</td></tr>"
                   for r in va["domain_sensitivity"])
    levels = "".join(f"<tr><td>{r['level_cm']:g}</td><td>{'Closed' if r['status']=='closed' else escape(r['status'].replace('_',' '))}</td>"
                     f"<td class='number'>{number(r['area_km2']) if r['area_km2'] is not None else '—'}</td>"
                     f"<td>{r['area_grid_relative_change']:.1%}</td></tr>" if r['area_grid_relative_change'] is not None else
                     f"<tr><td>{r['level_cm']:g}</td><td>{escape(r['status'].replace('_',' '))}</td><td>—</td><td>—</td></tr>"
                     for r in report["contours"]["levels"])
    warnings = "".join(f"<li>{escape(w)}</li>" for w in report["warnings"])
    sampling = va["conditional_sampling"]
    q = sampling["quantiles_km3"]
    interval = (f"The joint sampling grid gives an approximate conditional 95% interval of <b>{number(q['0.025'])}–{number(q['0.975'])} km³</b>. "
                f"{escape(sampling['interpretation'])} Quantile convergence has not been established; "
                "this is a sampling diagnostic, not a calibrated eruption-volume interval."
                if sampling["mean_grid_agrees_with_refined_integral"] else
                "The conditional sampling grid does not agree with the refined mean integral to within 1%. "
                "Its interval is retained in report.json for diagnosis and is not presented as a volume estimate here.")
    coverage = (f"Geographic holdouts: {nested['coverage95_observations']:.0%} coverage for nominal 95% observation intervals."
                if nested else "Independent geographic holdouts were not requested for this run.")
    bx = regional["bounds_km"]
    parent = '<p><a href="../index.html">← All deposits</a></p>' if report["settings"].get("atlas_parent") else ""
    body = f"""{parent}<div class="eyebrow">Tephra / Deposit atlas</div>
<h1>{escape(name)}</h1><p class="lead">A reproducible reconstruction from measured thicknesses, with closed isopachs and explicit limits on extrapolation.</p>
<span class="tag">{escape(report['model']['candidate']['tail'])} thinning</span><span class="tag">No invented observations</span><span class="tag warn">Regional volume · total unresolved</span>
<div class="stats"><div class="stat"><span class="label">Measurements used</span><strong>{report['input']['n_measurements']}</strong><small>{len(report['input']['ignored_pseudo_rows'])} legacy pseudo rows excluded</small></div>
<div class="stat"><span class="label">Expected regional bulk volume</span><strong>{number(regional['mean_km3'])} <small>km³</small></strong><small>{'Numerical refinement passed' if regional['converged'] else 'Numerical refinement unresolved'} · {regional['relative_change']:.2%} last change</small></div>
<div class="stat"><span class="label">Volume outside the data hull</span><strong>{hull:.0%}</strong><small>Of the expected volume in the stated region</small></div></div>
<div class="panel"><a href="isopachs.svg" title="Open scalable map"><img src="isopachs.svg" alt="{escape(name)} closed median isopachs with observations, data hull and volume region"></a>
<div class="caption">Solid contour sections lie inside the observation hull; dashed sections are extrapolated. Even a solid section can cross a data gap. Closed curves are properties of the fitted surface, not measured deposit boundaries.</div></div>
<div class="downloads"><a href="short_report.md">Short report · Markdown</a><a href="isopachs.pdf">Map · PDF</a><a href="isopachs.png">Map · PNG</a><a href="isopachs.svg">Map · SVG</a><a href="isopachs.geojson">Contours · GeoJSON</a><a href="isopach_areas.csv">Contour areas · CSV</a></div>
<h2>Volume, with a defined scope</h2><p>The headline integrates expected thickness over the ochre rectangle: easting {bx[0]:,.0f} to {bx[2]:,.0f} km and northing {bx[1]:,.0f} to {bx[3]:,.0f} km in the vent-centred equal-area projection. This region is selected from the observation locations, independently of the map extent.</p>
<p>The integral of the median thickness surface over the same region is <b>{number(regional['median_surface_km3'])} km³</b>. That is a different quantity from the median of total random volume. Neither quantity is dense-rock equivalent.</p>
<div class="table-wrap"><table><thead><tr><th>Integration region</th><th>Selected fit · km³</th><th>Other family · km³</th><th>Numerical refinement</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="note warn">Increasing both widths by 2× increases the area by 4×. These estimates show sensitivity to the integration region and thinning family; the range is not a confidence interval. Total erupted volume has not been established.</div>
<h2>Fit and numerical checks</h2><p>{coverage}</p><div class="panel"><img src="diagnostics.svg" alt="Contour area curve, volume-domain sensitivity and geographic holdout diagnostics"></div>
{empirical_section(report)}
<details><summary>Closed-contour areas and resolution checks</summary><p>Areas include all resolved components and subtract enclosed holes. Unclosed levels have no reported area. Small features are refined locally. The last column compares the final area grid with a grid of half the resolution; per-level scope is recorded in the CSV.</p>
<div class="table-wrap"><table><thead><tr><th>Thickness · cm</th><th>Status</th><th>Area · km²</th><th>Grid change</th></tr></thead><tbody>{levels}</tbody></table></div></details>
<details><summary>Conditional volume uncertainty</summary><p>{interval}</p></details>
<details open><summary>Interpretation and limitations</summary><ul>{warnings}</ul></details>
<div class="downloads"><a href="report.json">Full report · JSON</a><a href="volume_summary.csv">Volume estimates · CSV</a><a href="observations_used.csv">Measurements used · CSV</a></div>
<footer>Statistical method {report['method_version']}; reporting workflow {report['report_version']}. Input SHA-256: <code>{escape(report['input']['input_sha256'])}</code><br>Maps are generated offline. Coastlines: Natural Earth, public domain.</footer>"""
    (Path(output)/"index.html").write_text(page(name, body))


def publish_dataset(output, name, model, alternate, inverse, report, resolution=481,
                    levels=None, max_extent_km=6000.):
    output = Path(output)
    options = {} if levels is None else {"levels": levels}
    contours = build_contours(model, resolution=resolution, max_extent_km=max_extent_km, **options)
    report["contours"] = {k: contours[k] for k in ("bounds_km", "outer_level_cm", "resolution", "max_extent_km", "levels")}
    report["report_version"] = "0.5.0"
    report["dataset"] = name
    report["volume_analysis"] = volume_analysis(model, alternate, report["volume"]["bounds_km"], report["volume"])
    report["warnings"] = [w for w in report["warnings"] if not w.startswith("Doubling the integration grid")]
    if any(not row[which]["converged"] for row in report["volume_analysis"]["domain_sensitivity"]
           for which in ("selected", "alternative") if row[which]):
        report["warnings"].append("At least one volume integral did not meet the 0.5% refinement criterion; inspect volume_analysis before using it.")
    omitted = [r["level_cm"] for r in contours["levels"] if r["status"] == "extends_beyond_map"]
    if omitted:
        report["warnings"].append(f"Contours {omitted} cm extend beyond the reporting extent; no closure or area is fabricated for them.")
    inaccurate = [r["level_cm"] for r in contours["levels"]
                  if r["area_grid_relative_change"] is not None and r["area_grid_relative_change"] > .02]
    if inaccurate:
        report["warnings"].append(f"Contour areas {inaccurate} cm change by more than 2% between map grids; refine --map-grid for these small features.")
    report["warnings"].append("All volume estimates are bulk deposit volumes; no deposit-density conversion to DRE is available.")
    pd.DataFrame(contours["levels"]).to_csv(output/"isopach_areas.csv", index=False)
    (output/"isopachs.geojson").write_text(json.dumps(geographic_contours(contours, inverse), allow_nan=False)+"\n")
    data = model.data
    lon, lat = inverse.transform(*data.xy.T)
    pd.DataFrame({"source_row": data.row_ids, "Lon": lon, "Lat": lat,
                  "Kind": np.where(data.censored, "upper_limit", "measurement"),
                  "Thickness_cm": np.where(data.censored, np.nan, np.exp(data.log_value)),
                  "Limit_cm": np.where(data.censored, np.exp(data.log_value), np.nan),
                  "LogSD": data.log_sd}).to_csv(output/"observations_used.csv", index=False)
    rows = []
    for row in report["volume_analysis"]["domain_sensitivity"]:
        for label in ("selected", "alternative"):
            if row[label]:
                values = row[label]
                rows.append({"domain_factor": row["domain_factor"], "model": label,
                             **{k: values[k] for k in ("mean_km3", "median_surface_km3", "converged", "relative_change",
                                                       "fraction_expected_volume_outside_data_hull")},
                             **dict(zip(("xmin_km", "ymin_km", "xmax_km", "ymax_km"), values["bounds_km"]))})
    pd.DataFrame(rows).to_csv(output/"volume_summary.csv", index=False)
    save_isopach_map(output, name, model, report["input"], contours, report["volume_analysis"])
    save_diagnostics(output, name, report, contours, data)
    write_short_report(output, name, report)
    write_dataset_page(output, name, report)
    return report


def write_atlas_index(output, reports):
    cards, table, summary = [], [], []
    for slug, report in reports:
        name = report["dataset"]
        regional = report["volume_analysis"]["regional"]
        coverage = report["nested_validation"]["coverage95_observations"] if report["nested_validation"] else None
        outer = report["contours"]["outer_level_cm"]
        cards.append(f'''<article class="card"><a href="{escape(slug)}/index.html"><img src="{escape(slug)}/isopachs.png" alt="{escape(name)} isopach map" loading="lazy"></a><div class="body"><div class="eyebrow">{report['input']['n_measurements']} measurements · {report['model']['candidate']['tail']} fit</div><h2><a href="{escape(slug)}/index.html">{escape(name)} ↗</a></h2><p><b>{number(regional['mean_km3'])} km³</b> expected regional bulk volume</p><small>Closed outer contour: {outer:g} cm · {regional['fraction_expected_volume_outside_data_hull']:.0%} of regional volume outside data hull</small></div></article>''')
        table.append(f"<tr><td><a href='{escape(slug)}/index.html'>{escape(name)}</a></td><td>{report['input']['n_measurements']}</td><td class='number'>{number(regional['mean_km3'])}</td><td>{regional['fraction_expected_volume_outside_data_hull']:.0%}</td><td>{f'{coverage:.0%}' if coverage is not None else 'Not run'}</td></tr>")
        summary.append({"dataset": name, "measurements": report["input"]["n_measurements"],
                        "ignored_pseudo": len(report["input"]["ignored_pseudo_rows"]),
                        "selected_tail": report["model"]["candidate"]["tail"],
                        "expected_regional_bulk_km3": regional["mean_km3"],
                        "median_surface_regional_bulk_km3": regional["median_surface_km3"],
                        "volume_outside_hull_fraction": regional["fraction_expected_volume_outside_data_hull"],
                        "volume_grid_converged": regional["converged"],
                        "outer_closed_contour_cm": outer, "nested_observation_coverage95": coverage})
    body = f'''<div class="eyebrow">Tephra / Reproducible deposit atlas</div><h1>From measurements<br>to isopachs.</h1><p class="lead">{len(reports)} deposits. One statistical workflow. Explore closed thickness contours, the measurements behind them, and what the missing regions mean for volume.</p><div class="downloads"><a href="summary.csv">Compare deposits · CSV</a><a href="README.md">Reproduction and interpretation</a></div><div class="note">The original simple fit is retained. Legacy pseudo-points are excluded. All reported volumes refer to stated regions; none is claimed as total erupted volume.</div><div class="grid">{''.join(cards)}</div><div class="rule"></div><h2>Read the numbers with their support</h2><p class="muted">Regions differ between deposits. Volume outside the data hull measures reliance on extrapolation, not an error bound. Coverage describes held-out observations, not total-volume accuracy.</p><div class="table-wrap"><table><thead><tr><th>Deposit</th><th>Measurements</th><th>Regional bulk · km³</th><th>Volume outside hull</th><th>95% holdout coverage</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><footer>Directional thinning + spatial residuals. No historical wind reconstruction. Maps and downloadable data work offline.<br>Coastlines: Natural Earth, public domain. Full assumptions and code/input hashes are recorded in each report.</footer>'''
    output = Path(output)
    (output/"index.html").write_text(page("Tephra deposit overview", body))
    pd.DataFrame(summary).to_csv(output/"summary.csv", index=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    readme = """# Tephra deposit atlas

Read the individual README reports linked below; no local server is required.

Rebuild from the repository root with:

```sh
python -m src.probabilistic_isopach atlas --output results/atlas
```

Each deposit includes PNG/SVG/PDF maps, GeoJSON contour lines, enclosed-area
tables, volume estimates for three explicit regions and two thinning families,
measurements used, diagnostics, and a full reproducibility report.

The statistical method is recorded separately for every deposit. Dashed isopach sections are
outside the measurement convex hull. The map expands to show closed contours;
unresolved levels are listed, never joined along the frame. Contour areas
subtract holes and are checked against a coarser grid.

Headline volumes are expected **bulk deposit volume in a finite region**.
They are not DRE or total erupted volume. Integrals are refined until consecutive
mean- and median-surface estimates agree within 0.5%, or a failure is reported.
The median-surface integral is not the median of random total volume.
Conditional sampling intervals are secondary diagnostics; parameter uncertainty
and quantile-grid convergence are not established. Domain/family sensitivity is
reported separately and is not a confidence interval.

The original four input collections have not been independently re-audited.
YTT preserves the supplied audit and its qualifications. No new field
measurements or eruption-physics inputs were introduced. Existing nested
validation is reproduced, including unfavorable results.
"""
    if reports:
        map_grid = reports[0][1]["settings"]["map_grid"]
        method = reports[0][1]["settings"].get("method", "selected")
        readme = readme.replace("atlas --output results/atlas\n", f"atlas --output results/atlas --map-grid {map_grid} --method {method}\n")
    readme += "\n" + "\n".join(f"- [{r['dataset']}]({slug}/README.md)" for slug, r in reports) + "\n"
    (output/"README.md").write_text(readme)
