# Ambiguous thicknesses and interpreted bounds

[Assessment](README.md) · [All site decisions](site-evidence-decisions.csv) · [Selected input](model-selection.csv) · [Protocol](../../../docs/methods/sites-without-measured-thickness.md)

The principal input contains **43 reported thicknesses, 23 conditional lower bounds and one conditional upper bound**. An ambiguous total does not disqualify a separately documented primary component. The original 14 lower-bound decisions below are retained, with nine further minima documented below and in the final citation pass. The current 67-observation selection was frozen before refitting. The [preceding29-observation selection](history/2026-11-23-expanded-bounds/README.md) is retained. A smaller fitted volume or a smoother contour was not a selection criterion.

## Calculation and physical meaning

Let `h` be the preserved primary bulk thickness and `a=h/T` its effective ratio to original local airborne bulk thickness `T`. The principal lower bounds assume `0<a<=1`, so `T>=h`. A reported recovered range `[l,u]` then permits `T>=l`. Missing phases, erosion and compaction can reduce `a`; focusing or imported ash can make it exceed one. The primary-bed description supplies a reason for the in-place interpretation, **not a measured upper bound on focusing**. These are conditional physical constraints, not confidence limits. The coherent half-transfer test permits an assumed `a<=2`, giving `T>=h/2`; the factor2 is a stress assumption, not a geological measurement. The omission test removes all 23 minima and the Skull Creek upper bound. The weaker-transfer test also doubles the Skull Creek ceiling to 10 cm as a separate, explicitly hypothetical relaxation of the author estimate.

The thickness convention remains bulk centimetres. No universal decompaction correction is applied. Both range endpoints remain in the ledger; the current reader uses one endpoint per locality. Separate refits use the other endpoints where a source range is available. An upper endpoint additionally requires negligible loss and compaction, and is therefore less secure for original fallout. No midpoint, duplicated endpoint likelihood or invented measurement error is used.

| Evidence ID / locality | Principal bound (cm) | Reported evidence and interpretation |
|---|---:|---|
| B2020-046 Mount Bachelor | >=44 | Buckland2024 TableS1 row5 separates44 cm UPU from12 cm precursor NLPU and8 cm ash. Only the identified climactic primary component supplies this minimum. |
| B2024-UKR Upper Kirk Road | >=35 | TableS1 row14 gives35–40 cm UPU under20 cm reworked pumice/soil. Exclude the overburden; use35 once. |
| B2020-134 Portage Inlet | >=1.27 | Buckley and Willis1970 p.101 gives1/2–3/4 inch: multiply by2.54 to obtain1.27–1.905 cm. The <=1.905 alternative is a separate refit. |
| B2020-017 Independence Lake | >=2 | S1 row29 describes2–6 cm Mazama in retrieved cores. Bovet2003 corroborates coring, the event horizon and coherent ages; its accessible text does not reproduce that numerical range. Use the lower recovered endpoint with compilation-level thickness qualification. |
| B2020-019 Lake Cleveland | >=2 | S1 row31 gives the same 2–6 cm range. Bovet2003 corrects the setting: failed coring of the deepest basins led to sampling an eastern sub-basin about7 m deep. Core coordinates and the numerical range remain qualified. |
| B2020-021 Wildhorse Lake | >=2 | S1 row33 separates the earlier0.1 cm fall and later2 cm fall. Matz1987 Table6 row32 lists preserved C1=2 cm and C2=0.1 cm; its4 cm value is a correction. The bound assumes the later layer is climactic Mazama. |
| B2020-072 Lost Trail Pass | >=4.9 | S1 row84 identifies a4.9 cm primary basal component with seasonal pollen evidence. The7 cm compilation total and8 cm interval in another core remain unreconciled; neither is substituted for this component. |
| B2020-103 Wildcat Lake | >=1 | S1 row115 distinguishes1 cm primary ash by color, refractive index, stratigraphy and pollen. Matz Table6 row51 separates preserved C1=1 cm from C2<1 cm. The 146 cm mixed total is excluded. Original Blinman/Foit logs remain pending. |
| B2020-122 Newman Lake | >=15 | Matz Table6 row48 explicitly gives preserved15–30 cm, supporting a range interpretation of the compiler's22.5+/-7.5. No Gaussian uncertainty or wet-to-upland correction is inferred. |
| B2020-229 Jesmond Bog | >=0.5 | S1 row241 records5 mm before doubling to an upland equivalent. Matz Table6 row3 also records preserved 0.5 cm. The conversion is5/10=0.5 cm. |
| B2020-291 Lofty Lake | >=0.1 | S1 row303 identifies a1 mm bed at398 cm depth. Matz Table6 row2 distinguishes preserved 0.1 from corrected 0.2 cm; Buckland2022 also interprets at least0.1 cm. This is not a generic trace code. |
| W1993-CC-1A / section26 group | >=2 | Welcher1993 pp.51–52 describes2 cm basal fine ash under reworked UnitIV; purity and conformable contact favor primary airfall. CC-1A,1B,5 and6B share this section-level constraint. |
| W1993-CC-9 / section31 | >=2 | Same primary-component interpretation; preserve the thesis's inconsistent profile list rather than treating it as a second reading. |
| W1993-CC-11 / section6 group | >=2 | Same2 cm component; CC-11 and13 are represented once at section resolution. Repeated fine beds and11–235 cm fluvial totals are not summed. |

The Independence, Cleveland and Newman alternate endpoints are tested together as `range-upper` (<=6, <=6 and <=30 cm respectively); this is a grouped source interpretation, not an attribution of the change to one site. The Portage endpoint is tested separately. Williams Lake Fen's phase ambiguity has its own [omission test](WILLIAMS_FEN.md).

## Additional conditional minima

| Evidence ID | Lower endpoint (cm) | Source derivation |
|---|---:|---|
| B2023-GOLD |74|78 cm reported bed minus4 cm mixed basal material; target-phase membership of all remaining coarse ash is conditional.|
| W2017-LSUNRISE |3|3 cm recovered at the core base; incomplete penetration.|
| M2024-ROUNDWA |60|Approximate60 cm bed identified from age/stratigraphy; no imported thickening assumed.|
| W1995-TT06313 |2|211−209 cm discrete-bed interval; endpoint interpretation and primary marine deposition conditional.|
| M2014-CLELAND-F |4|Conservative lower endpoint of4–5 cm primary text/log disagreement; not a statistical interval.|
| MK2012-PATJENS |4|4 cm interpreted silicic bed; Mazama identity and primary fraction conditional.|

[Primary references, coordinates and caveats](FOLLOWUP_SEARCH.md) and the full ledger preserve the observed quantities separately from these transfers. New point beds also receive a lower-limit alternative. The weaker-identity omission includes the age/visual correlations; Gold Lake has its own phase/transfer omission. All 23 minima are omitted together and weakened coherently in the shared source tests. Dog Lake Oregon (≥12 cm), White Pine Marsh (≥25 cm) and Connley Cave 2 (≥28 cm) are derived from their primary descriptions in the [final citation review](LATE_SOURCE_REVIEW.md). The same review documents Skull Creek’s conditional ≤5 cm author estimate, the unresolved decimal typography, and the separate omission test. The 25 cm mixed Skull Creek deposit is not fitted as fallout.

## Sources, location and dependence

Matz's [1987 thesis text](https://studylib.net/doc/11985296/an-abstract-of-the-thesis-of), pp.92–94 Table6, was checked through publicly indexed text. Its table convention explicitly puts preserved wet values in parentheses and corrected values outside. This cross-check does not turn the thesis's literature compilation into an independent field measurement. The original site investigators remain identified in the ledger; source logs not obtained remain in the work queue. The Lofty occurrence (not its1 mm thickness) is independently corroborated by the [Lichti-Federovich1970 abstract](https://doi.org/10.1139/e70-089) at398 cm; the thickness still relies on the cited compilation trail. [Bovet, Bowerman and Cadol2003](https://keckgeology.org/wp-content/uploads/2024/07/bovetbowermancadol-2003.pdf), PDF pp.2–4, Table1 and Fig.3, supplies the lake context and age check. Direct download of these two web sources returned403; accessible indexed text was read, and no original PDF or HTML payload is claimed preserved for them.

Camp Creek's original AppendixA land descriptions were found before considering map digitization. The retained [official BLM polygon response](../../sources/mazama/camp-creek-blm-sections.json) locates three sections; [derived coordinates and full footprints](camp-creek-georeferencing.json) make the centroid calculation reproducible. A section centroid is not the surveyed core position. The maximum centroid-to-vertex distances describe the geometric footprint, not a survey error distribution. Four profiles in section26 and two in section6 are counted once at the available section resolution. All seven profile identities remain in the locality ledger. Their finer aliquot descriptions and relative distances remain recoverable from the preserved thesis; no coordinates are digitized from its figure. Predictions across the retained footprints are checked separately, and these section groups do not create seven independent regional observations.

Other inherited lake coordinates remain approximate, including the Lofty location discrepancy and the Independence group of lakes. The source table supplies locations; no unreported coordinate uncertainty is invented. The model-based checks cannot validate a wrong core identity or an assumed preservation factor.

## Decisions that remain unquantified

- Juke Box trench has a primary geochemical match and a preserved0–3 cm spatial range with an approximately2 cm mean. A zero at a local pinch-out does not imply absence of original fallout. The range supplies no strictly positive minimum across the trench, and the mean has no documented averaging footprint or preservation transfer. A3 cm upper bound on original fallout would additionally need a lower bound on retention. Keep the range and identity; neither endpoint presently constrains original local thickness at the compiler's unspecified sample point.
- Mica Creek's alluvial-fan/ditch and stump-associated accumulations, Calgary's discontinuous reworked material, Cypress's alluvial description and Brule's diffuse basal sand preserve useful component or host bounds. Their descriptions do not isolate a primary thickness or bound loss/import. Their existing preserved bounds remain in the ledger, without being relabelled as original fallout.
- Generic trace codes, Keephills' ash-rich host interval and cryptotephra counts lack a demonstrated transfer to original bulk thickness. A positive occurrence is retained; an analytical detection or sample width is not assigned as a thickness limit.
- Other unresolved composite phases, field-log conflicts and unlocated horizons retain site-specific missing terms in the decision table. No low-geometric-influence exemption is claimed merely from a similar whole-Earth volume.

## Scope of verification

All 293 compilation entries are reconciled, plus 166 additional records, leads and controls. **257 original entries remain at compilation-only verification**; other verification labels distinguish partial context checks from full thickness checks. The resulting dataset is a source-qualified conditional reference, not certification that every accepted value equals original deposition. Each fit uses the frozen common input hash, and all source tests retain their own inputs, fit warnings and numerical checks.
