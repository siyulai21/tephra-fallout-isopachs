# Eruption data and sources

The dataset catalogue pairs each accepted CSV with its source audit and model settings. Use the case home for observation counts and results. [Dataset catalogue](datasets.json) · [Accepted CSVs](eruptions/README.md) · [Eruption results](../eruptions/README.md)

The source audits explain how published observations became model inputs. An evidence ID identifies a source row, its reported units, any conversion and its use as a thickness measurement, a bound or a contextual record. Aso-4 and B-Tm each retain the original CSV alongside the reviewed version. The Mazama ledger covers 867 source records and documents how 271 localities enter the model. Tambora uses the supplied inclusion flags. [Source audits](audits/README.md) · [Source material](sources/README.md)

## CSV schema

The required columns are `Lon`, `Lat`, `Thickness`, `Vent Lon` and `Vent Lat`.

- Coordinates: WGS84 decimal degrees. Thickness: centimetres.
- Supply a consistent complete vent pair on at least one row.
- Optional `Pseudo`: blank/No/False/0 = observation; Yes/True/1 = excluded invented value.
- Optional `Kind`: `measurement` (default), `upper_limit` or `lower_limit`.
- Upper limits and lower bounds require documented, positive `Limit_cm`. A lower bound describes minimum recovered thickness in an incomplete deposit, not an exact measurement.
- Optional `LogSD`: known additional standard deviation of natural-log thickness. The fitted residual nugget is separate.

For a real site with a defensible interpreted bound, leave `Thickness` blank, set the appropriate `Kind` and enter the positive `Limit_cm`. Preserve its evidence ID, the source observation and the conversion in the audit. Keep unquantified occurrences in the site ledger. The accepted CSV supports a single one-sided bound per observation. The protocol describes how to preserve both endpoints of a range and compare them in separate fits. [Sites without measured thickness](../docs/methods/sites-without-measured-thickness.md)

Illustrative schema only:

```csv
Lon,Lat,Thickness,Vent Lon,Vent Lat,Pseudo,Kind,Limit_cm,LogSD
-122.37,43.05,850,-122.1,42.95,,measurement,,0.10
-124.19,42.03,,,,,upper_limit,0.5,0.20
-129.96,55.43,0.01,,,Yes,measurement,,
```

The upper-limit example is not geological evidence. Missing, eroded or unsampled deposits are not zero original deposition. Invalid coordinates, zero/missing measurement thicknesses, unsupported traces and ambiguous flags raise errors. Every run records the input SHA-256 and accepted CSV row identifiers.
