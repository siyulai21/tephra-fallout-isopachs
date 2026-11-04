> Historical selection; use the [history index](../README.md) for current file locations.

# Audited YTT thickness inputs

The [original workbook](../../YTT_thickness_audit.xlsx) is preserved byte-for-byte from the **Audit YTT Thickness Records** chat (2026-11-02). Its `Occurrence audit` sheet contains 88 records. The [selection manifest](selection_manifest.csv) preserves all 17 source columns, including evidence, references, URLs, coordinates, decisions and thickness qualifications, plus explicit model-row mappings. This import verifies the audit; it does not independently re-review the cited publications.

| Case | Records | Selection |
|---|---:|---|
| [Principal input](principal_45.csv) | 45 | All 43 nonblank audited `fit_cm` entries plus N001/N002, primary YTT airfall at Maninjau |
| [Without Maninjau](../2026-11-02/without_maninjau_43.csv) | 43 | Exclusion sensitivity; identical to the former main input |
| [Without MD161 additions](no_md161_40.csv) | 40 | Principal input minus N021–N025; both Maninjau sections retained |

**Selection revision, 2026-11-03:** Maninjau A40 (N001) and A64 (N002) now enter the principal dataset at the reported approximately **23 cm each**. The inherited audit identifies primary airfall beneath younger Maninjau flow deposits and cites Alloway et al. (2004), section 3.3 and Figs. 2–3. The [publisher abstract](https://doi.org/10.1016/j.epsl.2004.08.014) independently confirms two underlying tephra beds correlated with YTT. This inclusion is based on their geological interpretation, not on whether they improve a fitted score or target volume.

Their positions remain approximate user-workbook coordinates: A40 at 100.5377° E, −0.3362° N; A64 at 100.3013° E, −0.3157° N. These are not newly verified surveyed positions, and their displayed decimals do not imply that precision. Location uncertainty is not propagated in the current model intervals. The 50 km validation buffer limits leakage between these nearby sections; retain the without-Maninjau sensitivity until location QA improves. No coordinate distribution or thickness error bar is invented.

The original workbook's blank `fit_cm` values and original audit notes are preserved. Explicit principal-selection overrides, the revision rationale and source qualifications are recorded in `provenance.json`; the manifest adds principal and sensitivity memberships. Historical [38-record](../2026-11-02/no_md161_38.csv) and [45-record](../2026-11-02/with_maninjau_45.csv) inputs remain available to interpret earlier reports. The latter is byte-identical to the current principal input.

Huguangyan has no validated thickness and is excluded. The hypothetical 0.05 cm scenario is not a measurement. Malawi assumed and illustrative equivalent values are excluded. Chala remains gross preserved-deposit thickness, not solid-glass equivalent. RC14-37 uses observed 15 cm, not reconstructed approximately 20 cm. All other blank `fit_cm` records remain excluded.

Reported ranges can describe alternate contacts or local variation; they are retained as audit metadata and not converted into invented error bars. The model uses the audited working point values. No pseudo-points, zero-thickness constraints or undocumented detection limits are added. These qualifications limit the geological interpretation even when numerical calculations converge.

The common source reference is 98.8417° E, 2.6083° N, the [Smithsonian GVP Toba caldera coordinate](https://volcano.si.edu/volcano.cfm?vn=261090). It is not a uniquely established YTT vent. Changing the source position would be another sensitivity analysis.

[Provenance JSON](provenance.json) records input/workbook hashes and all case memberships. The source workbook SHA-256 is `19931ca2eb1b0ee1ae4cb77f41a2c014bf55bc7b71e1c70bad73bbe90a5f1628`.

Verify that all CSVs rebuild exactly from the workbook, with the documented Maninjau inclusion overrides:

```sh
python -c 'from benchmarks.ytt import verify_inputs; print(verify_inputs())'
```

Reproduce the complete calculations with `python -m benchmarks.ytt --output results/ytt`. See the [published report and method decision](../../../../../archive/reports/ytt/archive/2026-11-03-maninjau/README.md).
