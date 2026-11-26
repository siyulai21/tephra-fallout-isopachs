from pathlib import Path
import json
import subprocess
import time

home = Path('results/directional-geography-v6/final-checks')
modules = [
    'verify_btm_audit', 'verify_bearing_continuation', 'verify_aso_evidence',
    'verify_directional', 'verify_directional_ensemble', 'verify_azimuth_continuation',
    'verify_directional_multiscale', 'minimum_fit_results', 'minimum_fit_release --verify',
    'mazama_release_check',
]
records = []
for module in modules:
    args = module.split()
    start = time.monotonic()
    with (home / (args[0]+'.txt')).open('w') as out:
        run = subprocess.run(['.venv/bin/python', '-m', 'benchmarks.'+args[0], *args[1:]], stdout=out, stderr=subprocess.STDOUT)
    records.append(dict(command='python -m benchmarks.'+module, exit_code=run.returncode, elapsed_seconds=round(time.monotonic()-start, 1)))
    (home/'control-results.json').write_text(json.dumps(records, indent=2)+'\n')
    print(module, run.returncode, flush=True)
raise SystemExit(any(r['exit_code'] for r in records))
