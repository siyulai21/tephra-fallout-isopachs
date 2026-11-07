# B-Tm: recovered source table and observation meanings

[Geometry review](../../README.md) · [Accepted input](../../../../data/eruptions/B-Tm_tephra.csv) · [Two-site sensitivity](../../btm-input-sensitivity/README.md)

The published supporting workbook for [Yang et al. (2021)](https://doi.org/10.1007/s00445-021-01487-8) supplies **76 uniquely corresponding sites** in the current 84-observation B-Tm table. Every source coordinate agrees with its counterpart to the current table's 0.001-degree rounding. This recovers names, reference labels and observation qualifications; it is not an independent verification of all underlying publications or their surveyed locations.

The [original workbook](yang2021-thickness.xlsx) was downloaded from the [University of Cambridge repository](https://www.repository.cam.ac.uk/handle/1810/330433) and is preserved unchanged. The repository states a [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). Attribution: Qingyuan Yang and coauthors, *The Millennium Eruption of Changbaishan Tianchi Volcano is VEI 6, not 7* (2021), Supplementing dataset 1. The extracts add source-row identities and comparison fields; original source descriptions are retained.

## What the 76 matches reveal

| Published source meaning | Rows | Treatment in current accepted table |
|---|---:|---|
| Numeric thickness without a qualification in this table | 59 | All 59 numerical values agree |
| Thickness range | 13 | A single endpoint or midpoint is used |
| Thickness described as spots | 2 | A single thickness is used without that qualifier |
| Lake Suigetsu: less than 0.1 cm | 1 | CSV row 82 treats 0.1 cm as exact |
| Lake Kushu: no visible ash layer | 1 | CSV row 84 treats 0.1 cm as exact |

Kushu is discussed as cryptotephra in the supporting text. Presence of dispersed shards is not a measured 0.1 cm bulk layer, nor is it a zero deposit. Suigetsu supplies an explicit upper bound. The [separate calculation](../../btm-input-sensitivity/README.md) isolates these two meanings under the source-focused model: Suigetsu enters as an upper bound and Kushu contributes no quantitative thickness likelihood. It does not deny Kushu's reported tephra presence, invent a detection limit, or replace the accepted collection.

The thirteen ranges include 16–25 cm at `KH69-2-23` (current CSV row 7 uses 25), 3–5 cm at `St.6920` (row 26 uses 5), and 1–4 cm at `KH69-2-25` (row 32 uses 4). Other ranges have been replaced by midpoints. Their meaning may involve within-site variation, preservation or measurement uncertainty. The source papers need review before these can be assigned a common interval likelihood or standard deviation. No arbitrary numerical error has been derived from a range or quality label. The two spot occurrences also remain qualified rather than being silently discarded.

Eight accepted rows are absent from this distal compilation: six proximal entries and two other sites. Their [original row identities and values](unmatched-current-rows.csv) remain explicit. In particular, this comparison does **not** independently verify the nearest 150 cm observation used to diagnose B-Tm's previous displaced maximum. The vent coordinates and accepted inputs are unchanged.

## Research workbooks are not interchangeable with the published table

Two user-held workbooks, `B-Tm.xlsx` and `B-TmR.xlsx`, helped locate the source but differ from its original observation descriptions. The latter includes 89 positive entries and a separate 116-row zero table. Those additional entries have not been matched to reliable detection limits, chronology or preservation evidence and are not used in the present fits. Their displayed labels are not independently verified citations.

The published supporting text itself discusses marine preservation and reporting limitations when considering missing layered ash. Those records must not be converted into certain original zero fallout. The original pseudo-points already marked in the accepted CSV remain excluded. Recovering another zero table does not justify restoring subjective closing points.

## Inspect the evidence

- [All 76 source records](source-thickness-records.csv) and [their unique current-row crosswalk](source-crosswalk.csv)
- [Eight unmatched current rows](unmatched-current-rows.csv)
- [Hashes, download provenance, original workbook cells, counts and limitations](summary.json)
- [Read-only extraction](extraction.mjs.txt), [comparison script](audit.py.txt), and [artifact hashes](artifacts.json)

The crosswalk uses an explicit ±0.00050001° coordinate comparison to allow the supplied decimal rounding; this is not an inferred location-error distribution. The source sheet is `thickness_data!A2:F77`. The original supporting archive, its workbook member name and checksum are recorded, so the public source can be downloaded and verified independently. No input change or volume reduction is hidden in this provenance review.
