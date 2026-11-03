# YTT reviewed thickness inputs

The principal dataset contains **46 measured-thickness records**: the original 43 accepted observations, two qualified Maninjau primary beds, and **U1445B at 8 cm**. All 88 records were screened against the inherited audit. New primary-source checks focused on U1445B/U1448B and their IODP descriptions; this is not a claim that all cited original papers were independently re-audited.

The [original workbook](YTT_thickness_audit.xlsx) remains byte-for-byte unchanged. The [selection manifest](selection_manifest.csv) preserves all 17 source columns, including original blank `fit_cm` cells, and adds effective coordinates and model-row mappings. [Selection rules](selection_rules.json) contain the explicit overrides. [Record review](record_review.csv) gives a decision, reason, evidence scope and remaining action for every record.

| Current case | Records | Purpose |
|---|---:|---|
| [Principal](../../eruptions/YTT_tephra.csv) | 46 | Accepted observations, including Maninjau and U1445B |
| [Without Maninjau](sensitivity/without_maninjau.csv) | 44 | Tests approximate-position primary beds; retains U1445B |
| [Without U1445B](sensitivity/without_u1445b.csv) | 45 | Isolates the newly added hole; identical to the previous principal input |
| [Without MD161](sensitivity/without_md161.csv) | 41 | Removes five already accepted, bioturbated beds |
| [With U1448B](sensitivity/with_u1448b.csv) | 47 | Tests the disputed 20 cm interpretation; does not promote it |

## Why these records belong in the principal dataset

**Maninjau A40/A64 (N001/N002):** the inherited source audit identifies primary YTT airfall beneath younger Maninjau flow deposits, approximately 23 cm at each section. [Alloway et al. (2004)](https://doi.org/10.1016/j.epsl.2004.08.014) confirms the correlation of two underlying beds with YTT. Inclusion follows this geological interpretation, not a desired model score or volume. Coordinates remain approximate user-workbook values: A40 100.5377° E, −0.3362° N; A64 100.3013° E, −0.3157° N. Their displayed decimals do not establish surveyed accuracy. Position uncertainty is not propagated into the model intervals; the exclusion sensitivity remains useful.

**U1445B (N005):** [Kutterolf et al. (2023)](https://doi.org/10.1016/j.jvolgeores.2023.107879), Table 1, p. 18, identifies section `U1445B-2H-6_42–50` as YTT event 1 and reports 8 cm of fine ash with a horizontal base. The section interval independently spans 8 cm. The [IODP U1445 report](https://publications.iodp.org/proceedings/353/105/353_105.html), Table T1, supplies hole B coordinates: 17°44.7098′ N, 84°47.2498′ E. Decimal conversion is explicit in the selection rules; original rounded audit coordinates remain visible. The shipboard description gives approximately 10 cm for the A/B ash occurrence; this revision uses the later hole-specific tabulated 8 cm. The disturbed, inclined companion U1445A (T022) remains excluded, so U1445 contributes one site representative.

**MD161 N021–N025:** these five records already belonged to the original principal 43. Their exclusion sensitivity never meant they were absent from the principal calculation. Keep their gross preserved-bed thicknesses and bioturbation qualifications; no pure-glass density is inferred.

## Why other records remain held

**U1448B (N006)** is the closest additional candidate. Kutterolf et al., Table 1, p. 20, gives 20 cm for `U1448B-1H-4/5_136/–6` with an irregular base. The [IODP U1448 report](https://publications.iodp.org/proceedings/353/108/353_108.html) describes hole B ash from 5.53 to 6.09 m, a 56 cm interval. These are competing contact descriptions, not endpoints of a measurement-error distribution. The sensitivity uses the 2023 20 cm value and qualified approximate audit coordinates. Reconciling the core contacts is needed before principal inclusion.

The remaining held records have unresolved identity, reworking, disturbance, contact/location conflicts, missing primary measurement tables, or thicknesses that describe mixed sampling/occurrence intervals. SO93 compiled values are not promoted merely because the [Gasparotto et al. (2000) abstract](https://doi.org/10.1016/S0025-3227(99)00090-0) describes primary ash: the exact original measurement table was not retrieved. All individual reasons are in the [88-record review](record_review.csv).

Huguangyan has no validated thickness; hypothetical 0.05 cm is not a measurement. Malawi assumed/count-derived illustrative thickness is excluded. Chala remains gross preserved-deposit thickness. RC14-37 uses observed 15 cm, not reconstructed approximately 20 cm. A held record is not a non-detection or zero deposit.

## Reproducibility and limits

No pseudo-points, arbitrary measurement-error bars, range midpoints, spatial weights or new detection limits are added. Approximate locations and preservation uncertainty remain limitations. Nearby sections can be spatially dependent; the existing covariance model and geographic validation buffer address some of that dependence, not every geological source of correlation. The Toba source coordinate, 98.8417° E and 2.6083° N, is a caldera reference rather than a uniquely known YTT vent.

```sh
python -m benchmarks.ytt --prepare-only
python -c 'from benchmarks.ytt import verify_inputs; print(verify_inputs())'
python -m benchmarks.ytt --output results/ytt
```

Preparation deterministically rebuilds the principal CSV, four sensitivities, manifest and provenance from the unchanged workbook plus reviewed rules. Verification checks every generated byte, all 88 source rows and the historical input checksums. The benchmark uses a new output directory and runs the principal selected/stacked methods plus four selected-method sensitivities.

[Historical inputs](history/README.md) and [archived reports](../../../reports/ytt/archive/README.md) remain available. [Current calculations](../../../reports/ytt/README.md) contain maps, validation and regional volumes. The original workbook SHA-256 is `19931ca2eb1b0ee1ae4cb77f41a2c014bf55bc7b71e1c70bad73bbe90a5f1628`.
