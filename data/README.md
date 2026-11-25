# Eruption data and sources

The dataset catalogue pairs each accepted CSV with its source audit and model settings. Use the case home for observation counts and results. [Dataset catalogue](datasets.json) · [Accepted CSVs](eruptions/README.md) · [Eruption results](../eruptions/README.md)

The source audits explain how published observations became model inputs. An evidence ID identifies a source row, its reported units, any conversion and its use as a thickness measurement, a bound or a contextual record. Aso-4 and B-Tm each retain the original CSV alongside the reviewed version. The Mazama audit covers 872 locality and evidence records and documents 300 accepted localities. Tambora uses the supplied inclusion flags. [Source audits](audits/README.md) · [Source material](sources/README.md)


For eruptions with one-sided reported minima, `reported_file` identifies the geological table and `file` identifies the fitting table. Both retain the source row order. [Preparing source and fitting tables](../docs/methods/reported-minimum-fitting.md)

## CSV schema

The required columns are `Lon`, `Lat`, `Thickness`, `Vent Lon` and `Vent Lat`.

- Coordinates: WGS84 decimal degrees. Thickness: centimetres.
- Supply a consistent complete vent pair on at least one row.
- Optional `Pseudo`: blank/No/False/0 = observation; Yes/True/1 = excluded invented value.
- Set `Kind` to `measurement` (default), `upper_limit`, `lower_limit` or `interval`.
- Upper limits and lower bounds require documented, positive `Limit_cm`. A lower bound describes minimum recovered thickness in an incomplete deposit, not an exact measurement.
- Optional `LogSD`: known additional standard deviation of natural-log thickness. The fitted residual nugget is separate.

For a one-sided bound, leave `Thickness` blank and put the positive limit in `Limit_cm`. For a two-sided interval, leave both of those fields blank and enter positive, ordered `Lower_cm` and `Upper_cm` on one row. All five families fit and score the probability between its endpoints. Preserve the source range, units and transfer assumptions in the ledger, including for partial or mixed packages. [Sites without measured thickness](../docs/methods/sites-without-measured-thickness.md)

Illustrative schema only:

```csv
Lon,Lat,Thickness,Vent Lon,Vent Lat,Pseudo,Kind,Limit_cm,LogSD
-122.37,43.05,850,-122.1,42.95,,measurement,,0.10
-124.19,42.03,,,,,upper_limit,0.5,0.20
-129.96,55.43,0.01,,,Yes,measurement,,
```

The upper-limit example is not geological evidence. Missing, eroded or unsampled deposits are not zero original deposition. Invalid coordinates, zero/missing measurement thicknesses, unsupported traces and ambiguous flags raise errors. Every run records the input SHA-256 and accepted CSV row identifiers.
