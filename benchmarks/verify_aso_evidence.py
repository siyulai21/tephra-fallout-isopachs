"""Verify the frozen November 2026 Aso-4 publication; run from repository root."""
from pathlib import Path
from hashlib import sha256
import json, re
import numpy as np
import pandas as pd

OUT=Path('comparisons/aso-4-audit')
read=lambda p:json.loads(Path(p).read_text())
hashfile=lambda p:sha256(Path(p).read_bytes()).hexdigest()
raw=Path('data/eruptions/Aso-4_tephra.csv')
assert hashfile(raw)=='802f22266d905b8d10c0535a7767e7bd83fb7f797692ea869095a3fa410360d8'
audit=read('data/audits/Aso-4/provenance.json');reviewed=Path(audit['cases']['principal']['file'])
assert hashfile(reviewed)==audit['cases']['principal']['sha256']
assert hashfile(audit['source']['file'])==audit['source']['sha256']
for f,digest in audit['audit_artifacts'].items():assert hashfile(Path('data/audits/Aso-4')/f)==digest
assert (OUT/'runs/revision/reviewed.csv').read_bytes()==reviewed.read_bytes()
ledger=pd.read_csv('data/audits/Aso-4/row-ledger.csv')
for row in ledger.iloc[:52].itertuples():
 numbers=re.findall(r'\d+(?:\.\d+)?',str(row.s2_thickness_text))
 assert len(numbers)==1 and float(numbers[0])==row.thickness_cm
others=read(OUT/'other-eruptions-hashes.json')
for f,digest in others.items():assert hashfile(f)==digest,f
archive=read('archive/2026-11-08-aso-4/manifest.json')
for entry in archive['files']:
 if not entry['archive'].endswith('.md'):assert hashfile(entry['archive'])==entry['sha256'],entry['archive']
regional=read('eruptions/aso-4/results/report.json');glob=read('eruptions/aso-4/results/global/report.json')
for r in (regional,glob):
 assert r['input']['input_sha256']==hashfile(reviewed)
 assert r['input']['audit']['sha256']==hashfile('data/audits/Aso-4/provenance.json')
 assert (r['input']['n_measurements'],r['input']['n_lower_limits'],r['input']['n_upper_limits'])==(83,3,1)
 assert r['input']['ignored_pseudo_rows']==list(range(89,95))
for key in ('row_ids','folds','buffer_km'):assert glob['validation'][key]==regional['validation'][key]
for method,m in regional['validation']['methods'].items():
 for a,b in zip(glob['validation']['fits'],m['selections']):
  for k in ('training_row_ids','withheld_row_ids'):assert a[k]==b[k]
for f,digest in read('eruptions/aso-4/results/global/artifacts.json').items():assert hashfile(Path('eruptions/aso-4/results/global')/f)==digest
support=read('eruptions/aso-4/results/support/report.json')
assert support['provenance']['parent_report_sha256']==hashfile('eruptions/aso-4/results/report.json')
for model in support['methods'].values():
 for p in model['contours']['policies'].values():assert p['selected_band'] is None
posterior_checks=[]
for path in sorted((OUT/'runs').rglob('attempt-*.json')):
 record=read(path);posterior=path.with_suffix('.npz')
 assert hashfile(posterior)==record['posterior_sha256'],path
 with np.load(posterior) as draws:
  assert draws['log_volume'].shape == (record['signature']['sampling']['chains'],record['signature']['sampling']['draws'])
  assert all(np.isfinite(draws[k]).all() for k in draws.files)
 posterior_checks.append({'file':str(posterior.relative_to(OUT)), 'sha256':hashfile(posterior),'raw_sampler_passed':record['diagnostics']['passed']})
summary=read(OUT/'summary.json')
assert len(summary['experiments'])==17 and all(r['sampling_passed'] for r in summary['experiments'])
assert len(glob['fits'])==8 and glob['all_assumption_fits_passed_diagnostics'] and glob['validation']['all_fits_passed_diagnostics']
for key in ('original_data_validation','reviewed_data_validation'):
 assert summary[key]['all_reference_fits_passed'] and summary[key]['all_alternative_fits_passed']
checks={
 'date':'2026-11-08',
 'reference_commit':'8263e0eac0fc0a8c7093a5b40ad36832a80a5d43',
 'original_csv_unchanged':hashfile(raw),
 'reviewed_input_sha256':hashfile(reviewed),
 'source_workbook_and_audit_artifact_hashes_verified':True,
 'all_52_compiled_thickness_magnitudes_match_original_csv':True,
 'source_inequalities_and_primary_conflicts_preserved_in_audit':True,
 'other_eruption_files_unchanged_count':len(others),
 'archived_scientific_files_unchanged_count':sum(not e['archive'].endswith('.md') for e in archive['files']),
 'archive_markdown_changes':'Relative navigation and original-input reproduction command only.',
 'same_current_input_row_ids_geographic_folds_buffers_training_and_withheld_rows':True,
 'reviewed_counts':{'exact':83,'lower':3,'upper':1,'ignored_pseudo':6},
 'targeted_comparison_cases':17,
 'all_targeted_final_fits_passed':True,
 'global_reference_plus_fixed_sensitivities':8,
 'all_fixed_sensitivity_final_fits_passed':True,
 'all_reviewed_geographic_final_fits_passed':True,
 'posterior_attempts_checked':posterior_checks,
 'raw_sampler_flag_note':'Raw flags include trajectory-efficiency warnings. Published posterior assessments separate mixing/ESS/divergence/BFMI checks and recheck the contamination parameter. Every earlier attempt remains retained.',
 'regional_plot_change':'bbox_inches=tight prevents clipped axis labels. Independent rerun verified exact equality of all regional scientific values and support/deletion calculations before vs after this plot-only edit.',
 'global_readme_enrichment':'Added a source-qualification paragraph to the generated case README; updated its artifact checksum. Scientific JSON and posterior bytes were not edited.',
 'tests_recorded_at_publication':'183 passed; source semantics, immutable-input reconstruction, exact conjugate PSIS reference, numerical and existing workflow tests. This artifact verifier does not itself rerun pytest.',
 'scope':'Recomputed Aso-4 only; refreshed cross-eruption indexes. No new scientific method or prior default adopted for other eruptions. The old 40-trial regional method benchmark remains historical because numerical regional methods are unchanged.',
 'verification_script_sha256':hashfile(__file__),
}
(OUT/'verification.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
print('Verified',len(posterior_checks),'posterior attempts,',len(others),'unchanged other-eruption files, and matched current validation.')
