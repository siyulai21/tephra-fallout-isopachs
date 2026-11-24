# Mazama audit data dictionary

Historical stage. The [current source decisions](principal-ranges/locality-decisions.csv) give the fitted reading and field qualification for each record in the 300-locality release.

[Assessment](README.md) · [Revised ledger](revised-locality-ledger.csv)

The revised ledger is an evidence inventory, not the strict model-input schema. Empty numeric cells mean **unknown or not applicable**, never zero. Descriptive unknowns are explicit. Thickness units are centimetres; latitude/longitude are decimal degrees with western longitudes negative. No source datum is assumed known merely because coordinates are printed as decimal degrees.

| Field group | Meaning |
|---|---|
| `record_id`, `parent_record_id` | Stable audit identifiers. `B2020-NNN` preserves published site number. Parent links indicate a related compilation entry, not proof of duplicate measurements. Additional records can share a core catchment, section or locality. |
| `source_excel_row` | Physical Excel row, including headings. For S1, row = site number +12. |
| `latitude`, `longitude` | Declared locality coordinates. Usually inherited from S1; primary DMS and authoritative-inventory corrections are explicitly labeled. The Moss catalogue row retains its general lake coordinate while the principal selection records the central-core coordinate from Egan2016 p.305. Missing additional coordinates are not guessed from a place name. |
| `coordinate_basis`, `coordinate_quality`, `location_description` | Provenance, limitations and native PLSS/DMS descriptions. Long decimal strings do not imply high positional accuracy. Native Montana Georeferenced sections and later digitized localities include source-specific coordinate evidence. |
| `primary_reference` | Original references or source chain. An inherited citation does not imply the paper was read. |
| `evidence_source`, `evidence_locator` | Actually inspected material and row/page/table/section. Local originals are listed in the source manifest; web-only reads and failed downloads are identified. |
| `verification` | What was checked: compilation only, full relevant primary passage, original field description, abstract, photograph caption, source reinterpretation or partial check. No aggregate “verified” Boolean hides these differences. |
| `identity`, `identity_quality` | Basis and confidence qualification for attribution to climactic Mazama, including possible precursor confusion and tentative signals. No new geochemical classification was calculated. |
| `local_age`, `age_basis` | Reported age/chronology and whether independent, imported, conventional radiocarbon, calibrated, or ice-layer counted. No recalibration was performed. Unknown ages remain unknown. |
| `deposit_class`, `setting` | Observed or inherited depositional context. `compiler_primary/secondary/both` is the S1 judgment, not an independent facies determination. |
| `compiler_thickness_cm` | Raw numeric or text summary from S1; retained even if rejected as a quantity. Consult the untouched raw extraction for all original metadata. |
| `quantity_kind` | Whether the quantity is a compilation point, measured preserved bed, component, approximate value, historical correction, conflict, range, bound, trace, cryptotephra or tentative signal. |
| `observed_point_cm` | Reported point value under the stated `quantity_kind` and verification. A filled cell does not mean accepted primary fallout. Historical interpretive values can remain here with their explicit label. |
| `observed_min_cm`, `observed_max_cm` | Descriptive range endpoints. Can represent spatial variation, multiple cores or compiler arithmetic. They are not automatically strict intervals or an error distribution. Approximate endpoints remain qualified in text. |
| `bound_relation`, `bound_cm`, `bound_target` | Explicit inequality and its target quantity. Current inequalities concern preserved deposits, not unconditional original fallout. The strict model input does not automatically ingest this table. |
| `interval_meaning` | Derivation and scope of a range/bound, including conversions. One inch =2.54 cm; one foot =30.48 cm. Source-rounded metric values are retained rather than given spurious precision. |
| `core_depth_or_sample_interval` | Depth of the horizon or analyzed sediment/ice interval. **Never interpret this field as ash thickness.** |
| `fallout_model_status` | Candidate, hold, occurrence-only or exclusion decision with a specific reason. `hold` means unresolved, not disproven. |
| `accepted_for_model` | True for the62 principal fitting records; this selection does not upgrade their evidence quality. Historical acceptance is documented separately by the crosswalk. |
| `principal_kind`, `principal_thickness_cm`, `principal_limit_cm`, `principal_decision`, `principal_assumption` | Accepted fitting quantity and its transfer assumptions, separate from the observed source quantity. Exact selected coordinates and row identities are in `model-selection.csv`. |
| `thickness_quality`, `caveats` | Measurement, preservation, phase, dependency and source limitations. No invented `LogSD`, density or recovery correction. |
| `source_thickness_notes`, `source_description` | Original compilation narrative retained alongside the audit. It may disagree with the reviewed quantities; such disagreement is intentional and documented. |

## Other tables

- `buckland2020-293-localities.csv` and `buckland2020-cells.json` preserve all fifteen original columns plus Excel row number. Original `-`, nulls, spellings and codes are retained. JSON null and empty CSV cell represent missing source cells.
- `supplied-row-crosswalk.csv` uses **one-based data row** and **one-based physical CSV line**; line = row +1. Do not confuse these with Excel rows or published site IDs. Coordinate displacements use a mean-radius spherical great-circle calculation (6371.0088 km) for audit diagnostics, not a datum correction.
- `bounds-and-ranges.csv`, `additional-localities.csv`, and `occurrences-and-traces.csv` are filtered views of the master ledger. They are not independent observations and should not be concatenated back into it.
- `review-decisions.json` holds source-specific judgments overriding clearly labeled compilation defaults. `additional-records.json` contains manually transcribed additional records. The builder validates record identities and absence of numeric invented cryptotephra quantities.
- `compilation-bibliography.csv` transcribes the published reference list. Bibliographic errors in that list are not silently repaired. Modern references are carried on individual additional records and in the source manifest.

## Interpretation rules

A positive trace is evidence for presence at the sampled location under its attribution assumptions, but lacks a numerical thickness limit unless the source supplies a justified detection/quantification threshold. A patch's lateral disappearance or a sampled interval with no visible ash is not evidence of zero original deposition. A sedimentary package can both gain and lose ash; calling it reworked does not by itself prove either an upper or a lower bound on original fallout.

Bed intervals, core depths, sampled thicknesses, analytical shard concentrations and original atmospheric fallout are separate quantities. The same separation applies to whole-package thickness and phase-specific measurements, and to a local age estimate versus an eruption age used to construct that local chronology.

`regional_description_not_point` identifies a regional quantity that cannot substantiate a measurement at the compiled locality. For Carbon River, the historical numeric value remains only in `compiler_thickness_cm` and the raw catalogue.

The principal reference contains 43 point values, 23 lower limits and one upper limit, with blank `LogSD`. Source scenarios test new-record omission, point-to-lower-limit transfer, weaker identities, Yellowstone clustering and deformation/reporting alternatives, coherent weaker bounds, the Portage/Independence/Cleveland/Newman alternate endpoints, the final five-site addition and Skull Creek alone. A half-transfer factor is a declared hypothetical stress test, not a new empirical correction.
