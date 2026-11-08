# Accepted eruption inputs

[Data guide and CSV format](../README.md) · [Source audits](../audits/README.md) · [Reports](../../eruptions/README.md)

These are the CSVs read by the named dataset profiles in [datasets.json](../datasets.json). A familiar filename outside this folder does not automatically inherit a dataset's profile.

| Input | Report | Provenance |
|---|---|---|
| [Minoan](Minoan_tephra.csv) | [Report](../../eruptions/minoan/README.md) | [Source-table audit](../audits/Minoan/README.md); 92 measurements and nine lower bounds |
| [YTT](YTT_tephra.csv) | [Report](../../eruptions/ytt/README.md) | [Reviewed selection](../audits/YTT/README.md); 46 measurements |
| [Mazama](Mazama_tephra.csv) | [Report](../../eruptions/mazama/README.md) | Supplied flags; 45 accepted measurements, seven excluded pseudo rows |
| [Tambora](Tambora_tephra.csv) | [Report](../../eruptions/tambora/README.md) | Supplied flags; 26 accepted measurements, two excluded pseudo rows |
| [Aso-4](Aso-4_reviewed.csv) | [Report](../../eruptions/aso-4/README.md) | [Source-qualified audit](../audits/Aso-4/README.md); 83 exact, three lower and one upper bound; six excluded pseudo rows |
| [B-Tm](B-Tm_tephra.csv) | [Report](../../eruptions/b-tm/README.md) | Supplied flags; 84 accepted measurements, seven excluded pseudo rows |

The four originally supplied CSVs retain their original bytes. Aso-4's revised input has a separate filename; its source conflict and preservation assumptions remain explicit. Supplied flags alone are not an independent re-audit of every field source. Do not replace blanks or missing deposits with invented zero thicknesses.
