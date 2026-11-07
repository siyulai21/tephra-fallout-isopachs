# Targeted Aso-4 input review

[Source-geometry review](../README.md) · [Accepted input](../../../data/eruptions/Aso-4_tephra.csv)

The geometry comparison preserves the accepted input. The following source and coordinate checks do **not** constitute a complete audit of its 87 observations. In particular, 52 rows have no reference labels. No general numerical reduction of the input thicknesses is justified by these findings.

## Row 55: an eroded layer treated as exact

CSV row 55, labelled `Shibata2025`, is at 43.759167° N, 144.312500° E and contains 7 cm. These coordinates agree within the input rounding with Loc.1 in [Shibata et al. (2025), sections 3.2 and 4.1](https://doi.org/10.1186/s40623-025-02230-9). The paper describes a partly eroded ash layer ranging from 4 to 7 cm there. The input is therefore a preserved maximum, not a demonstrated exact original depositional thickness.

The separate sensitivity uses 7 cm as a lower bound on original deposit thickness, retaining the same 87 source-row identities and all other entries. It does not guess an erosion correction or turn the within-outcrop range into an invented standard deviation. [Sensitivity result and numerical checks](../input-sensitivity/README.md). The original-data geometry comparisons remain distinguishable from this change in observation meaning.

## Nearby records require provenance checks

A screening inventory of all accepted pairs separated by less than 1 km on WGS84 is in [aso-4-nearby-pairs.csv](aso-4-nearby-pairs.csv). The 1 km radius is an explicit audit search setting, not a fitting length scale or a rule for declaring duplicate data. Distance alone does not establish dependence or duplication.

| Original CSV rows | Separation | Supplied thicknesses | Issue to resolve |
|---|---:|---|---|
| 60 and 69 | 31 m | 10 and 10 cm | Different reference labels may describe the same exposure; original source correspondence is not established |
| 66, 79 and 80 | Pairwise 359–801 m | 31, 30 and 30 cm | Possible related exposures or repeated reporting in the same basin |
| 52 and 61 | 498 m | 3 and 10 cm | Local preservation/definition differences or source correspondence need inspection |
| 13 and 29 | 777 m | 5 and 3 cm | Neither record supplies a reference label |

All remain in the preserved geometry comparison. No pair has been automatically merged or discarded. The buffered geographic holdouts keep these nearby observations out of a heldout site's immediate training neighborhood, but that does not resolve possible dependence in the full-data likelihood or make the records independently audited.

## Recovered source workbook: observation meaning matters

[Rougier et al. (2022), section 4](https://doi.org/10.1093/gji/ggac131) describe 109 original measurements, nine pseudo-measurements, and 11 zero-thickness sites: marine holes with complete recovery of their upper 50 m but no reported tephra layers there. These recorded absences are distinct from the authors' invented boundary points. The current accepted collection has 87 positive measurements and no non-detection records; it is not a replication of that dataset.

A read-only inspection of the user-held `20210805_Aso4_Tephra.xlsx` recovered **98 thickness entries and 11 named non-detection sites**. That arithmetic agrees with 109 original records, but identity to the exact journal analysis input has not been established. The [workbook review](workbook-review/README.md) preserves the source hash, original sheet-row references, scientific data extracts and candidate correspondences. The eleven coordinates come from `DEFNO!A2:D12`, not from invented boundary points or the article's aggregate count.

The workbook's own key states that 0.5 cm can mean a trace or unspecified thickness with confirmed presence. It also warns that an unreported layer in the non-detection sheet could have been destroyed by bioturbation/coring, or simply not reported. These are not exact thicknesses or calibrated detection limits. Chronological coverage, preservation and reporting probability still need verification before assigning a likelihood. No recovered entry has been silently added to the present fits, and its effect on volume has not been calculated.

The coordinate comparison finds 48 current rows with a same-thickness candidate within 1 km. Those are candidate correspondences, not 48 independently verified source records. Several close sites have different thicknesses or multiple holes. A systematic coordinate offset in many `Machida85` candidates is consistent with a datum-conversion hypothesis, which is explicitly tested and left unadopted. Details and every candidate are retained in the linked review.

## Reproduce the source sensitivity

```sh
python -m benchmarks.aso_input_review --output results/aso-input-review
```

The command verifies the accepted-file checksum and the source site's coordinates before changing observation semantics in a separate derived CSV. It fits the preceding displaced geometry with unchanged priors. It does not select between geometric families or rerun geographic prediction; it isolates this one input interpretation.
