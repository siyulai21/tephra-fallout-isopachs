# Aso-4 source audit · 8 November 2026

[Eruption and current results](../../../eruptions/aso-4/README.md) · [Experiments and interpretation](../../../comparisons/aso-4-audit/README.md) · [Machine-readable provenance](provenance.json)

The principal calculation now uses a **source-qualified reference scenario**, with two corrected cruise coordinates and four inequality likelihoods. It is not a claim that every thickness is pristine atmospheric fallout. The [original CSV](../../eruptions/Aso-4_tephra.csv) remains unchanged; the [reviewed CSV](../../eruptions/Aso-4_reviewed.csv) preserves its order and row identities. All six legacy pseudo rows remain excluded from fitting.

## What was traced

Takarada & Hoshizumi's published [Table S2](https://doi.org/10.3389/feart.2020.00170.s002) contains 71 sites. Exactly 52 quantitative entries match the first 52 observations in our file by the recorded three-decimal coordinate rounding. The other 19 entries are `pod`, `tr` or `?`; they do not supply numerical thicknesses or detection limits. We do not turn them into 0.5 cm measurements, zero deposits, or censoring thresholds. The remaining 35 quantitative records in our CSV have separately supplied citations; this audit independently checked the Shibata record but does not certify all 35 primary sources.

- [Original published workbook](Takarada2020-TableS2.xlsx), reproduced under **CC BY 4.0**, attributed to Takarada & Hoshizumi (2020), Frontiers in Earth Science. No edits to the workbook.
- [Cell extraction](Takarada2020-TableS2-values.json): `Sheet1!B1:H76`.
- [Original row-to-source ledger](row-ledger.csv), [reviewed decisions and coordinates](record-review.csv), and [19 unquantified occurrences](s2-unquantified-presences.csv).
- [Source URLs, pages and downloaded-file hashes](sources.json).

Table S2 explicitly warns that source coordinate datums differ. Matching rounded coordinates establishes lineage, not datum accuracy. No undocumented datum conversion was applied.

## Corrections and qualifications

CSV row numbers below include the header. They are preserved in every fit and exclusion test.

| CSV row / site | Supplied value | Reference-scenario treatment | Evidence and limitation |
|---|---|---|---|
| 5 · MR00-K05 PC1 | 30.5° N, 147.5° E; 7 cm | **34.5° N**, same longitude and thickness | Yamamoto & Kanamatsu (2001), p.56 and cruise maps; independently corroborated by Yamauchi & Yamamoto (2001), p.42. About 444 km of coordinate displacement. |
| 8 · MR00-K05 PC2 | 40° N, 140° E; 6 cm | Same latitude; **146° E**, same thickness | Same cruise records; about 512 km displacement. |
| 43 · Machida site 5b | 10 cm exact | Upper bound 10 cm | Machida et al. (1985), Table 1, p.53 gives `<10`; S2 omitted the inequality. |
| 52 · Takano core | 3 cm exact | Lower bound 3 cm | Nagahashi et al. (2007), Table 1, p.308 gives `>3.0`. The table uses TKN2003 whereas S2 calls the core TKN-2004; retain this naming discrepancy. |
| 53 · Biwa B43-1 | 6.7 cm exact | Lower bound 6.7 cm **in this scenario** | S2 F31 gives `>6.7`, but Danhara et al. (2010), Table 4, p.108 gives **6.5–7 cm**. S2 also cites Takemura & Yokoyama (1989), not independently checked here. This conflict is unresolved. |
| 55 · Shibata Loc.1 | 7 cm exact | Lower bound 7 cm **under an erosion interpretation** | Shibata et al. (2025) describes 4–7 cm and partial erosion. This assumes no thickening through redeposition; the amount lost is unknown. |

Yamamoto & Aoki (2002), Tables 1–2, p.31, gives ash intervals 215.5–222.5 cm below seafloor in PC1 and 1148.5–1154.5 cm in PC2: thicknesses **7 and 6 cm**, respectively. Their p.32 correlation identifies both as Aso-4. Their abstract itself repeats PC1 latitude 30°30′; the correction follows the earlier cruise description, station maps and independent same-cruise study rather than silently ignoring this conflict.

The Biwa interpretation is a source choice, not an invented measurement error. Separate experiments retain 6.7 cm or use the primary table's 6.5 and 7 cm endpoints. Those endpoint fits are sensitivities, **not** rigorous bounds on the fitted total or a distribution over the reported range. The resulting volume is not used to decide which source is correct.

## Influential records are not automatically errors

Machida et al. (1985), Table 1 labels four proximal records uncertain (`afa?`, or `pfl or afa`: CSV 4, 6, 14, 20). We retain them in the reference scenario and explicitly test their omission. They are not proven flow deposits. CSV 37, KH76-2-7, is a valid **15 cm interval** (395–410 cm below seafloor), not a 395 cm deposit. CSV 49, KH82-4-23, is a 1 cm interval (847–848 cm). Deleting these records in separate experiments is an influence check, not a new exclusion decision.

The terrestrial Hokkaido thickness maximum cannot be dismissed as marine-current redistribution. The older field table and [Shibata et al. (2025)](https://doi.org/10.1186/s40623-025-02230-9) support relatively thick distal Aso-4 there; the latter describes preservation beneath Kutcharo deposits. We retain these data.

CSV 31 (SKY-S2) remains unresolved: S2 and the user's later compilation disagree in latitude and thickness coding. The original S2-backed quantitative entry is retained pending a primary-core check. Nearby observations can also be dependent; close pairs are flagged in the review ledger without automatically merging separate cores or interpreting coordinate proximity as duplication.

## Marine transport and the literature

[Machida & Arai (1988), pp.237 and 241](https://doi.org/10.4116/jaqua.26.3_227) tentatively associates unexpectedly few detected tephra layers near the Nankai Trough with the Kuroshio and identifies current effects on marine tephra distribution as a research problem. These passages do **not** establish a numerical Kuroshio correction for Aso-4 or demonstrate that our thick offshore records were enriched by it. The distinction matters: scarcity can reflect non-preservation as well as transport, and an unrecorded horizon is not a measured zero.

[Ikehara (2015), p.183](https://doi.org/10.18940/kazan.60.2_181) describes high glass abundance in modern seabed sediment along the Kuroshio as indirect evidence for transport. His [English review](https://doi.org/10.1186/s40645-015-0068-z) also discusses primary fallout, remobilization, bioturbation and current effects. This is later support for the general mechanism, not an Aso-4-specific quantitative attribution. [Takarada & Hoshizumi (2020)](https://doi.org/10.3389/feart.2020.00170) itself notes that submarine currents during settling can affect the fallout thickness distribution.

The published [Vineberg et al. (2025)](https://doi.org/10.1016/j.jvolgeores.2025.108436), checked alongside the earlier preprint in the user's project folder, strengthens proximal–distal eruption correlations using glass chemistry and Lake Suigetsu. It does not supply an Aso-4 Kuroshio redistribution correction. Targeted English/Japanese searches and these reference trails did not locate a study quantifying that correction. This is a bounded literature search, not proof that none exists.

The implemented response is a source-blind setting-removal stress test: remove all 28 entries explicitly marked submarine in S2, including thin and thick layers, under two separately stated geometry priors. The retained 59 include 35 records without independently audited settings, so we do **not** call them a verified terrestrial-only dataset. The test also changes spatial coverage; any volume difference cannot uniquely identify marine transport. See the [observation-model implications](../../../docs/methods/marine-observations.md).

No source volume, modern current path, arbitrary marine weight or invented geological uncertainty is used to tune the estimates.
