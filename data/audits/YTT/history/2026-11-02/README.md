> Historical selection; use the [history index](../README.md) for current file locations.

# Audited YTT thickness inputs

The [original workbook](../../YTT_thickness_audit.xlsx) is preserved byte-for-byte from the **Audit YTT Thickness Records** chat (2026-11-02). Its `Occurrence audit` sheet contains 88 records. The [selection manifest](selection_manifest.csv) preserves all 17 source columns, including evidence, references, URLs, coordinates, decisions and thickness qualifications, plus explicit model-row mappings. This import verifies the audit; it does not independently re-review the cited publications.

| Case | Records | Selection |
|---|---:|---|
| [Main input](without_maninjau_43.csv) | 43 | All nonblank audited `fit_cm` entries, including qualified retained beds |
| [Without MD161 additions](no_md161_38.csv) | 38 | Main input minus N021–N025, five bioturbated Mahanadi beds |
| [With Maninjau](with_maninjau_45.csv) | 45 | Main input plus N001/N002 at reported 23 cm and approximate coordinates; sensitivity only |

Huguangyan has no validated thickness and is excluded. The hypothetical 0.05 cm scenario is not a measurement. Malawi assumed and illustrative equivalent values are excluded. Chala remains gross preserved-deposit thickness, not solid-glass equivalent. RC14-37 uses observed 15 cm, not reconstructed approximately 20 cm. All other blank `fit_cm` records remain excluded.

Reported ranges can describe alternate contacts or local variation; they are retained as audit metadata and not converted into invented error bars. The model uses the audited working point values. No pseudo-points, zero-thickness constraints or undocumented detection limits are added. These qualifications limit the geological interpretation even when numerical calculations converge.

The common source reference is 98.8417° E, 2.6083° N, the [Smithsonian GVP Toba caldera coordinate](https://volcano.si.edu/volcano.cfm?vn=261090). It is not a uniquely established YTT vent. Changing the source position would be another sensitivity analysis.

[Provenance JSON](provenance.json) records input/workbook hashes and all case memberships. The source workbook SHA-256 is `19931ca2eb1b0ee1ae4cb77f41a2c014bf55bc7b71e1c70bad73bbe90a5f1628`.

Verify that all CSVs rebuild exactly from the workbook, with the documented sensitivity overrides only:

```sh
python -c 'from benchmarks.ytt import verify_inputs; print(verify_inputs())'
```

Reproduce the complete calculations with `python -m benchmarks.ytt --output results/ytt`. See the [published report and method decision](https://github.com/siyulai21/tephra-fallout-isopachs/blob/41b8f2f28ac8b1c877810b782005ff243e101c63/archive/reports/ytt/archive/2026-11-02/README.md).
