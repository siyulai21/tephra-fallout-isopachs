# Historical YTT selections

These inputs document earlier decisions. Current accepted inputs and sensitivities are one level above. Historical provenance JSON and manifests preserve the paths and hashes used at the time; those path strings are historical records, not current file locations.

- `2026-11-02/`: original audit provenance and manifest; `without_maninjau_43.csv` is the former principal input; `no_md161_38.csv` and `with_maninjau_45.csv` are the original sensitivity inputs.
- `2026-11-03-maninjau/`: the intermediate 45-record principal input, its provenance and manifest, and the 40-record MD161 exclusion input. Its 43-record exclusion input is the same original CSV stored in the preceding directory.

The source workbook is preserved once in the parent directory. Current `verify_inputs()` checks the original three input hashes against their historical provenance. Earlier README files are retained as historical narratives; use this index for relocated inputs and the [report archive](../../../../reports/ytt/archive/README.md) for calculations.
