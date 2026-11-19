# Eruption data and sources

The dataset catalogue pairs each accepted CSV with its source audit and model settings. Use the case home for observation counts and results. [Dataset catalogue](datasets.json) · [Accepted CSVs](eruptions/README.md) · [Eruption results](../eruptions/README.md)

The source audits explain how published observations became model inputs. An evidence ID identifies a source row, its reported units, any conversion and its use as a thickness measurement, a bound or a contextual record. Aso-4 and B-Tm each retain the original CSV alongside the reviewed version. Mazama and Tambora use the supplied inclusion flags. [Source audits](audits/README.md) · [Source material](sources/README.md)

## CSV schema

The required columns are `Lon`, `Lat`, `Thickness`, `Vent Lon` and `Vent Lat`.

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
