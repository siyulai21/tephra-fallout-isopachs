from pathlib import Path
from hashlib import sha256
import subprocess
import json

path = Path('benchmarks/published-evidence.sha256')
old = {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in path.read_text().splitlines()}
changed = set(subprocess.check_output(['git', 'diff', '--name-only']).decode().splitlines())
extra = set(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard']).decode().splitlines())
allowed = ('archive/2026-11-25-directional-v5/', 'archive/2026-11-26-directional-development/', 'comparisons/', 'eruptions/')
names = set(old) | {n for n in changed if n.startswith(('eruptions/', 'comparisons/', 'data/', 'docs/'))} | {n for n in extra if n.startswith(allowed)}
stale = json.loads(Path('results/directional-geography-v6/preexisting-stale-inventory.json').read_text())
base_hashes = {r['path']: r['unchanged_at_base'] for r in stale}
updates = {}
for name in sorted(names):
    digest = sha256(Path(name).read_bytes()).hexdigest()
    if name in old and digest != old[name]:
        assert name in changed | extra or digest == base_hashes.get(name), name
        updates[name] = dict(previous=old[name], current=digest)
    old[name] = digest
path.write_text(''.join(f'{h}  {n}\n' for n, h in sorted(old.items())))
receipt = Path('results/directional-geography-v6/inventory-updates.json')
history = json.loads(receipt.read_text()) if receipt.exists() else []
history.append(updates)
receipt.write_text(json.dumps(history, indent=2)+'\n')
print(len(old), 'published files;', len(updates), 'existing entries refreshed')
