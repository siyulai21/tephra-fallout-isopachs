"""Compare registered whole-deposit estimands using the principal nested corridor."""
import numpy as np

from .nested_summary import ROOT, comparison_plot, read, table
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.models.nested.reach import PROTOCOL


def build(output, peer_rows):
    rows, reports = [], []
    for record in sorted(datasets(), key=lambda item: item["slug"]):
        slug = record["slug"]
        home = ROOT/"eruptions"/slug/"results"
        corridor = read(home/"nested-isopachs/report.json")
        broad = read(home/"nested-whole-area/report.json")
        peers = [r for r in peer_rows if r["slug"] == slug and r["method"] in ("Directional", "Spherical")]
        if (len(peers) != 2 or corridor["protocol"] != PROTOCOL or corridor["model_role"] != "principal"
                or any(r["input_sha256"] != corridor["input_sha256"] for r in peers)
                or broad["input"]["input_sha256"] != corridor["input_sha256"]):
            raise ValueError(f"Principal and comparison inputs differ for {slug}")
        for current, source in zip(corridor["folds"], read(home/"nested-whole-area/geographic.json")["folds"], strict=True):
            if any(current[k] != source["identity"][k] for k in ("training_rows", "test_rows")):
                raise ValueError(f"Unmatched partitions for {slug} fold {current['fold']}")
        lo, hi = corridor["conditional_volume_interval95_km3"]
        rows.extend(peers)
        rows.append(dict(eruption=record["name"], slug=slug, method="Nested corridor", lower95_km3=lo,
                         median_map_km3=read(home/"nested-isopachs/deposit.json")["median_map_volume_km3"], upper95_km3=hi, equal_region_log_score=corridor["geographic_score"],
                         input_sha256=corridor["input_sha256"]))
        reports.append((slug, corridor, broad))
    table(output/"cases.csv", rows)
    comparison_plot(rows, output/"volumes.png", nested_family="Nested corridor")
    lines = ["# Global fallout volume from nested dispersal corridors", "",
             "The corridor volume integrates the median-thickness map over Earth. The plot shows the posterior uncertainty in complete-deposit volume under each global model.", "",
             "![Directional, spherical and principal nested median-map integrals with conditional 95% volume intervals](volumes.png)", "",
             "| Eruption | Median-thickness map (km³) | Conditional 95% interval (km³) | Whole-area median-map integral (km³) | Corridor geographic score ↓ | Withheld exact readings inside 95% | Full-data checks | Geographic fits passing |",
             "|---|---:|---:|---:|---:|---:|---|---:|"]
    for slug, j, broad in reports:
        lo, hi = j["conditional_volume_interval95_km3"]
        mapped = read(ROOT/"eruptions"/slug/"results/nested-isopachs/deposit.json")
        c = j["exact_site_coverage95"]
        warnings = [name for name, passed in j["numerics"].items() if not passed]
        status = "passed" if not warnings else "**warning** ("+", ".join(warnings)+")"
        lines.append(f"| [{j['name']}](../../eruptions/{slug}/results/nested-isopachs/README.md) | **{mapped['median_map_volume_km3']:,.3g}** | {lo:,.1f}–{hi:,.1f} | "
                     f"{read(ROOT/'eruptions'/slug/'results/nested-whole-area/median-field.json')['median_map_volume_km3']:,.1f} | {j['geographic_score']:.3f} | {c['covered']}/{c['exact_sites']} | "
                     f"{status} | {sum(f['passed'] for f in j['folds'])}/{len(j['folds'])} |")
    scored = [j for _, j, _ in reports]
    bands = sum(bool(j["supported_band_policies"]) for j in scored)
    lines += ["", f"All three global families use identical buffered geographic holdouts. Across these {len(scored)} eruptions, each given equal weight, the mean negative log predictive score is "
              f"{np.mean([j['geographic_score'] for j in scored]):.3f} for the corridor and "
              f"{np.mean([j['reference_geographic_score'] for j in scored]):.3f} for the whole-area sensitivity. Lower scores favor a model’s predictions. "
              f"A complete band passed at least one nested support policy in {bands} of {len(scored)} eruptions. [Three-family estimates and scores](cases.csv) · [Matched regional replay](regional-replay.csv)", "",
              "## What the observations constrain", "",
              "Local contour fragments drawn solid have exact measurements on both sides and at least 90% posterior probability of crossing in a retained triangle. Gaps and sectors without observations retain conditional predictions. The common prior gives 25% probability to broad continuation. The data can update those odds. Each case reports sensitivity to alternative prior odds.", "",
              "| Eruption | Median volume fraction in retained triangles | Median fraction beyond the farthest exact site | Independent map integral from the fitted posterior (km³) |",
              "|---|---:|---:|---:|"]
    for _, j, _ in reports:
        inside, far, _ = j["allocation_fraction"]
        lines.append(f"| {j['name']} | {inside[1]:.1%} | {far[1]:.1%} | {j['median_map_volume_km3']:,.1f} |")
    lines += ["", "The distance and triangle diagnostics use different spatial definitions. Sector budgets, support rejections and the full-compass maps are in the case reports.", "",
              "The separate Calbuco campaign and the withheld B-Tm archive bounds favor the corridor in those tests. Withheld Aso-4 and Pinatubo marine observations favor the broader continuation. These checks bear directly on whether a mapped deposit reached the sampled sites. Ancient total volumes remain conditional on thickness in unobserved regions. [Field-survey and mapped-contour checks](../nested-reach/field-surveys/README.md) · [Matched reach tests and areas by bearing](../nested-reach/README.md)", "",
              "The registered Pinatubo case uses 194 observations. The 137-row development snapshot retains its original result and is outside the registered-case averages. [Audited case](../../eruptions/pinatubo-1991-layer-c/README.md) · [Earlier snapshot](pinatubo-preview/README.md)", "",
              "[Whole-area sensitivity, earlier profiles and their validation](whole-area-study.md) · [Corridor model and reproduction](../../docs/methods/nested-reach.md) · [Previous release](../../archive/2026-11-21-corridor-principal/README.md)"]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    return rows
