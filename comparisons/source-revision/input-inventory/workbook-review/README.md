# Aso-4 source-workbook comparison

[Targeted input review](../README.md) · [Accepted observations](../../../../data/eruptions/Aso-4_tephra.csv)

This is a read-only comparison with the user-held [2021 source workbook](https://docs.google.com/spreadsheets/d/1aYtW7v6AOdAyALz_ZWbQzetms28sIfXO/edit). It improves the provenance inventory without claiming a completed audit of every primary paper. **No accepted coordinate, thickness, bound, reference label or fitting input changed.**

The workbook contains 98 rows in `Aso_Tephra!A2:I99` and 11 reported absences in `DEFNO!A2:D12`. Their sum is consistent with the 109 original records described by [Rougier et al. (2022)](https://doi.org/10.1093/gji/ggac131), but the exact journal analysis input has not been verified against this workbook. The current accepted collection contains 87 positive observations and includes later sources; it is a different collection.

## Distinct observation meanings

| Evidence in the workbook | What it establishes | What must not be assumed |
|---|---|---|
| `Key!B4`: some 0.5 cm entries code a trace or unknown thickness with confirmed presence | Presence was reported | Exact 0.5 cm thickness, or a 0.5 cm detection limit |
| `Key!B8:B9`: identification and thickness quality are separate | The compiler distinguished correlation confidence from thickness reliability | A numerical measurement error implied by an A/B/C letter |
| `Key!B13`: complete upper-50-m recovery without a reported layer, with explicit preservation/reporting caveats | A reported non-detection | Zero original fallout, complete age coverage, or perfect detection |

The thickness-quality counts are 62 A, 15 B and 21 C; 18 source entries have the value 0.5 cm. A is a statement that a source supplies thickness, not proof of undisturbed primary deposition. The extracted scientific rows retain quality letters and notes. They are audit evidence and are **not additional training data**.

Using these non-detections statistically would require a detection/preservation model. For example, if `q` is the probability that a layer survives and is reported, and `d(T)` is detection probability at original thickness `T`, a simple non-detection likelihood is `1 − q d(T)`. Neither `q` nor a calibrated `d(T)` is supplied here. This equation identifies the missing assumptions; it is not a fitted model or a justification for choosing numbers to reduce the volume. Explicit thickness bounds, when actually supplied by a source, can instead enter the existing censored likelihood.

## Candidate correspondences, not automatic replacements

All coordinate pairs within **1 km** are retained in [candidate-crosswalk.csv](candidate-crosswalk.csv), using WGS84 geodesic distance. This radius is an audit search setting, not a duplicate rule or model length scale. The direct comparison temporarily treats both coordinate columns as WGS84; datum uncertainty is discussed below. It finds 57 candidate pairs for 52 accepted rows. Of those, 48 accepted rows have a same-thickness candidate. Nearby alternatives are retained rather than choosing the most convenient match.

| Current CSV row | Current thickness | Workbook candidate and original sheet row | Issue |
|---|---:|---|---|
| 29 | 3 cm | `Core_C9001C`, row 43, 5.5 cm, about 44 m away | Same-site/source-version correspondence needs review |
| 54 | 6 cm | `U1436A`, row 80, 5 cm; `U1436B`, row 81, 2.5 cm; both about 3 m away | Multiple holes and layer definitions prevent a coordinate-only replacement |
| 56 | 0.7 cm | `Ge99-10`, row 45, 1 cm, same coordinates to supplied precision | Thickness rounding, layer definition or source version remains unresolved |

Among the same-thickness `Machida85` candidates, 32 initially differ by more than 200 m. A fixed Tokyo-to-WGS84 (108) transformation brings all 32 within 75 m, while worsening another already-close candidate. This is evidence **consistent with a datum-conversion explanation**, not proof that the workbook used Tokyo coordinates. The original datum and source-table positions were not independently verified, and no transformation was applied to accepted data. The exact tested pipeline and every transformed distance are preserved. A uniform coordinate correction would be unjustified.

The original publication is [Machida, Arai and Momose (1985)](https://doi.org/10.18940/kazanc.30.2_49). A historical datum difference is plausible given Japan's changes in geodetic reference systems; [GSI describes those systems](https://www.gsi.go.jp/ENGLISH/page_e30030.html). The numeric match above is our own diagnostic calculation, not a finding attributed to those sources.

## Inspect and reproduce

- [Source thickness entries with original sheet rows](source-thickness-records.csv)
- [Eleven reported non-detections with original sheet rows](source-nondetections.csv)
- [Every candidate correspondence](candidate-crosswalk.csv)
- [Source hash, ranges, settings, counts and limitations](summary.json)
- [Read-only extraction script](extraction.mjs.txt) and [comparison script](audit.py.txt)
- [Checksums](artifacts.json)

The extraction reads the original workbook through the spreadsheet runtime without exporting or changing it. The comparison consumes its values and the unchanged accepted CSV; `pyproj` supplies the geodesic and explicitly hypothetical datum calculation. The original workbook remains in the local research cache and the linked Drive file. Reproducing the extraction requires access to that original and matching its recorded checksum. The published extracts preserve the numerical evidence; the workbook key is summarized above with exact cell references.
