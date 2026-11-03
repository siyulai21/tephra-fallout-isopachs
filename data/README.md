# Input data

`datasets.json` is the dataset catalogue. The four original CSVs in `eruptions/` were moved from `examples/Data/` **without changing their contents**. YTT was subsequently added from the supplied audited workbook.

| File | Measurements | Excluded pseudo rows |
|---|---:|---:|
| `Mazama_tephra.csv` | 45 | 7 |
| `Tambora_tephra.csv` | 26 | 2 |
| `Aso-4_tephra.csv` | 87 | 6 |
| `B-Tm_tephra.csv` | 84 | 7 |
| `YTT_tephra.csv` | 43 | 0 |
| `Minoan_tephra.csv` | 92 exact + 9 lower bounds | 0 |

For the original four datasets, counts reflect supplied flags, not an independent audit of field publications. YTT preserves the supplied audit selections and qualifications; [read the audit provenance](audits/YTT/README.md). Minoan combines all fallout fans using the [documented source decisions](audits/Minoan/README.md). None of these datasets contains explicitly typed upper limits. Unnamed extra columns are ignored and reported. Historical maps in `docs/archive/original-isopachs/` are not fitting inputs.

## CSV schema

Required: `Lon`, `Lat`, `Thickness`, `Vent Lon`, `Vent Lat`.

- Coordinates: WGS84 decimal degrees. Thickness: centimetres.
- Supply a consistent complete vent pair on at least one row.
- Optional `Pseudo`: blank/No/False/0 = observation; Yes/True/1 = excluded invented value.
- Optional `Kind`: `measurement` (default), `upper_limit` or `lower_limit`.
- Upper limits and lower bounds require documented, positive `Limit_cm`. A lower bound describes minimum recovered thickness in an incomplete deposit, not an exact measurement.
- Optional `LogSD`: known additional standard deviation of natural-log thickness. The fitted residual nugget is separate.

Illustrative schema only:

```csv
Lon,Lat,Thickness,Vent Lon,Vent Lat,Pseudo,Kind,Limit_cm,LogSD
-122.37,43.05,850,-122.1,42.95,,measurement,,0.10
-124.19,42.03,,,,,upper_limit,0.5,0.20
-129.96,55.43,0.01,,,Yes,measurement,,
```

The upper-limit example is not geological evidence. Missing, eroded or unsampled deposits are not zero original deposition. Invalid coordinates, zero/missing measurement thicknesses, unsupported traces and ambiguous flags raise errors. Every run records the input SHA-256 and accepted CSV row identifiers.
