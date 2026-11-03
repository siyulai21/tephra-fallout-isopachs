"""Human-readable reports and GIS exports, independent of fitting decisions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contours import build_contours, geographic_contours
from .figures import save_isopach_map, save_diagnostics
from .volume import volume_analysis


def number(value):
    return f"{value:,.1f}" if value >= 10 else f"{value:,.2f}"


def method_description(report):
    if "stacking" in report["validation"]:
        return "Spatial holdouts fitted an experimental mixture of exponential and power-law thinning candidates. Mixture quantiles and expected thickness are computed directly; whole-field draws preserve both spatial covariance and model disagreement."
    return f"Spatial validation selected the {report['model']['candidate']['tail']} thinning family within the directional-trend-plus-spatial-residual model."


def write_short_report(output, name, report):
    volume = report["volume_analysis"]
    regional = volume["regional"]
    extended = volume["domain_sensitivity"][-1]["selected"]
    nested = report["nested_validation"]
    closed = [r["level_cm"] for r in report["contours"]["levels"] if r["status"] == "closed"]
    omitted = [r["level_cm"] for r in report["contours"]["levels"] if r["status"] == "extends_beyond_map"]
    coverage = (f"Nested geographic holdouts covered {nested['coverage95_observations']:.0%} of exact measurements with nominal 95% intervals; "
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
    globe = report.get("global_fallout")
    global_summary = ""
    if globe:
        primary = globe['selected']
        alternative = globe['alternative']
        global_summary = ("## Primary result: complete-Earth fallout\n\n"
            f"**Expected global bulk fallout: {number(primary['mean_km3'])} km³.** "
            "The integral covers the complete WGS84 surface, independently of the display extent or diagnostic rectangle. "
            f"{'Numerical refinement and angular-phase checks passed.' if primary['converged'] else 'Numerical convergence FAILED; do not treat this as a resolved numerical estimate.'}\n\n"
            f"The integral of the global median thickness surface is {number(primary['median_surface_km3'])} km³. "
            f"{primary['fraction_beyond_farthest_observation']:.1%} of expected volume lies beyond the farthest sampled source-distance "
            f"({primary['farthest_observation_authalic_arc_km']:,.0f} km on the authalic sphere); "
            f"{primary['fraction_on_opposite_hemisphere']:.1%} lies on the opposite hemisphere. "
            "These fractions describe extrapolation, not measured deposition.\n\n")
        if alternative:
            global_summary += (f"The other fitted thinning family gives **{number(alternative['mean_km3'])} km³** globally. "
                               "This comparison tests tail assumptions; it is not a confidence interval.\n\n")
        global_summary += ("The result is conditional on the fitted thickness model and its global extrapolation. "
                           "No global probability interval is estimated. The regional simulation intervals below or in JSON do not apply globally. "
                           "It is bulk fallout volume; DRE and other eruption products require separate evidence. "
                           "[Global numerical record](global_volume.json) · [Global volume table](global_volume.csv)\n\n"
                           "## Historical regional diagnostic\n\n")
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

The analysis used **{report['input']['n_measurements']} thickness measurements** and {report['input']['n_upper_limits']} documented upper limits and {report['input'].get('n_lower_limits', 0)} lower bounds, excluding {len(report['input']['ignored_pseudo_rows'])} legacy pseudo rows. {method_description(report)}

{global_summary}
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

The regional table contains finite-region **bulk deposit volumes**, not total erupted volumes or DRE. The median-surface integral is not the median of total random volume. {region_description} {'Numerical convergence passed the 0.5% criterion.' if regional['converged'] else 'Numerical convergence did not meet the 0.5% criterion; this estimate needs refinement.'} Unsampled sectors and distant tails remain assumption-dependent. Uncertainty model: {report['model']['uncertainty'].rstrip('.')}. Fitted-parameter uncertainty is not integrated. {provenance_note}

[Numerical report](report.json) · [Map PDF](isopachs.pdf) · [Volume table](volume_summary.csv) · [Contours](isopachs.geojson)
"""
    (Path(output)/"short_report.md").write_text(text)
    figures = "\n![Median isopachs with observations and extrapolation](isopachs.png)\n\n![Geographic holdouts and numerical diagnostics](diagnostics.png)\n"
    (Path(output)/"README.md").write_text(text+figures)


def publish_dataset(output, name, model, alternate, inverse, report, resolution=481,
                    levels=None, max_extent_km=6000.):
    output = Path(output)
    options = {} if levels is None else {"levels": levels}
    contours = build_contours(model, resolution=resolution, max_extent_km=max_extent_km, **options)
    report["contours"] = {k: contours[k] for k in ("bounds_km", "outer_level_cm", "resolution", "max_extent_km", "levels")}
    report["report_version"] = "0.7.0"
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
    report["warnings"].append("All volume estimates are bulk deposit volumes; no deposit-density conversion to DRE is applied by this workflow.")
    pd.DataFrame(contours["levels"]).to_csv(output/"isopach_areas.csv", index=False)
    (output/"isopachs.geojson").write_text(json.dumps(geographic_contours(contours, inverse), allow_nan=False)+"\n")
    data = model.data
    lon, lat = inverse.transform(*data.xy.T)
    pd.DataFrame({"source_row": data.row_ids, "Lon": lon, "Lat": lat,
                  "Kind": np.where(data.lower_censored, "lower_limit", np.where(data.censored, "upper_limit", "measurement")),
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
    if 'global_fallout' in report:
        (output/'global_volume.json').write_text(json.dumps(report['global_fallout'], indent=2, allow_nan=False)+'\n')
        pd.DataFrame([{'model':key, **{k:value[k] for k in ('mean_km3','median_surface_km3','converged',
            'fraction_beyond_farthest_observation','fraction_on_opposite_hemisphere')}}
            for key,value in report['global_fallout'].items() if key in ('selected','alternative') and value]).to_csv(output/'global_volume.csv',index=False)
    save_isopach_map(output, name, model, report["input"], contours, report["volume_analysis"])
    save_diagnostics(output, name, report, contours, data)
    write_short_report(output, name, report)
    return report
