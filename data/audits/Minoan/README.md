# Minoan combined-fallout audit

[Source workbook and attribution](../../sources/minoan/README.md) · [Full 105-row manifest](selection_manifest.csv) · [Provenance](provenance.json) · [Calculation report](../../../eruptions/minoan/README.md)

The target is **total primary fallout from all fans together**, including Plinian and co-ignimbrite/dispersed fallout. A site contributes one total observation, not one training row per phase. Identified turbidites are removed. No separate fan fit, hand-drawn isopach, pseudo-point or invented non-detection enters the model.

For a complete record, total fallout = reported total − identified numeric turbidite thickness. A blank turbidite cell means no flow unit was identified in this compilation, not independent proof of absence. Where a total has no reliable phase partition, its numeric total is retained without inventing the missing split. For example, POS513-20 contributes 49 − 15 = 34 cm, and POS513-27 contributes 25 − 9 = 16 cm.

The main input contains **92 exact records and nine lower-bound records at 101 distinct locations**. Every source row remains in the manifest, including excluded rows.

| Source qualification | Treatment |
|---|---|
| Nine totals begin with `>` | Lower-bound likelihood on the sum of recovered numeric fallout components; never an exact measurement |
| POS513-25: total `>40`, Plinian 40, co-ignimbrite 6 | Minimum recovered fallout 46 cm; uncertainty above this value remains |
| Map ID 9, POS513-17: deposit not reached beneath Kolumbo | Exclude; missing thickness is not zero |
| Map ID 83: total 12, components 2 + 8 | Exclude from main; separate 10 cm and 12 cm sensitivity cases |
| Map ID 85: total 15, components 2 + 3 | Exclude from main; separate 5 cm and 15 cm sensitivity cases |
| Map IDs 88/89, MD04-2788/MD04-2760 | Identical supplied coordinates and 0.1 cm totals; use one location constraint (88), preserve both labels |
| Map ID 63, MAR03-28 | Retain explicit total 2 cm; component bounds `<1` and `<2` do not redefine the total |
| Map ID 24, POS513-41 | Retain workbook 32 cm; also test omission because Supplementary Note 1 describes a 66 cm sequence |
| Black Sea 0.1 cm records | Use nominal published totals; their original measurement/detection provenance is not independently re-audited |

The earlier PANGAEA catalogue is only a cross-check. It contains differences in layer definitions, thicknesses and core completeness. It is not combined with the 2023 table, which would duplicate observations. No error SD is inferred from disagreement between publications.

| Input | Exact thicknesses | Lower bounds | Purpose |
|---|---:|---:|---|
| [Main](../../eruptions/Minoan_tephra.csv) | 92 | 9 | All accepted fallout fans together |
| [Exact only](exact_only.csv) | 92 | 0 | Sensitivity to incomplete cores |
| [Component sums](component_conflicts.csv) | 94 | 9 | Include conflicting rows using their component sums |
| [Reported totals](total_conflicts.csv) | 94 | 9 | Include conflicting rows using the total column |
| [Without POS513-41](without_pos513_41.csv) | 91 | 9 | Sensitivity to the workbook/sequence-description mismatch |

Coordinates and units follow the source. The vent reference is 25.396° E, 36.404° N. The original Minoan vent position remains uncertain. [GVP Santorini](https://volcano.si.edu/volcano.cfm?vn=212040)

Verify the workbook, row manifest and five reviewed input tables with `python -m benchmarks.minoan`. Add `--prepare-only` to rebuild them. Run `tephra-isopachs compare Minoan --output results/new-minoan` for all four fits. The published case combines Plinian and co-ignimbrite fallout in one deposit. [Current Minoan maps and results](../../../eruptions/minoan/README.md)
