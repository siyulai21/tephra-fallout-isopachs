# Accepted eruption inputs

[Data guide and CSV format](../README.md) · [Source audits](../audits/README.md) · [Reports](../../reports/README.md)

These are the CSVs read by the named dataset profiles in [datasets.json](../datasets.json). A familiar filename outside this folder does not automatically inherit a dataset's profile.

| Input | Report | Provenance |
|---|---|---|
| [Minoan](Minoan_tephra.csv) | [Report](../../reports/minoan/README.md) | [Source-table audit](../audits/Minoan/README.md); 92 measurements and nine lower bounds |
| [YTT](YTT_tephra.csv) | [Report](../../reports/ytt/README.md) | [Reviewed selection](../audits/YTT/README.md); 46 measurements |
| [Mazama](Mazama_tephra.csv) | [Report](../../reports/mazama/README.md) | Supplied flags; 45 accepted measurements, seven excluded pseudo rows |
| [Tambora](Tambora_tephra.csv) | [Report](../../reports/tambora/README.md) | Supplied flags; 26 accepted measurements, two excluded pseudo rows |
| [Aso-4](Aso-4_tephra.csv) | [Report](../../reports/aso-4/README.md) | Supplied flags; 87 accepted measurements, six excluded pseudo rows |
| [B-Tm](B-Tm_tephra.csv) | [Report](../../reports/b-tm/README.md) | Supplied flags; 84 accepted measurements, seven excluded pseudo rows |

The four originally supplied CSVs retain their original bytes. Their flags are not an independent re-audit of every field source. Do not replace blanks or missing deposits with invented zero thicknesses. Selection sensitivities belong in the audit folders, not silently in these principal inputs.
