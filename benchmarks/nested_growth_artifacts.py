"""Verify and retain a computation artifact before using any returned fit packet."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
from zipfile import ZipFile

from src.empirical_isopach.models.nested.footprints.continuous import engine
from src.empirical_isopach.models.nested.study import digest


def install(packet, metadata, scratch, archive):
    record=json.loads(metadata.read_text())
    expected=record['digest'].removeprefix('sha256:')
    assert digest(packet)==expected and packet.stat().st_size==record['size_in_bytes']
    runs=json.loads(Path('archive/2026-11-25-nested-growth/validation-runs.json').read_text())
    run=runs[str(record['workflow_run']['id'])]
    assert record['workflow_run']['head_sha']==run['commit']
    assert 'only_artifacts' not in run or record['name'] in run['only_artifacts']
    exclusion=run.get('excluded_artifacts',{}).get(record['name'])
    with ZipFile(packet) as zipped:
        names=zipped.namelist()
        assert len(names)==len(set(names))
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        provenance=json.loads(zipped.read('provenance.json'))
        assert provenance['commit']==record['workflow_run']['head_sha']
        assert int(provenance['run_id'])==record['workflow_run']['id']
        assert provenance['engine_sha256']==engine()
        versions=json.loads(Path('archive/2026-11-25-nested-growth/development/run-environment.json').read_text())['packages']
        assert all(versions[k]==v for k,v in provenance['packages'].items())
        files={}
        for info in zipped.infolist():
            if info.is_dir(): continue
            assert not stat.S_ISLNK(info.external_attr>>16)
            payload=zipped.read(info.filename)
            files[info.filename]=sha256(payload).hexdigest()
            relative=Path(info.filename)
            if exclusion or relative.parts[0] not in ('known','quadrants','surveys','continuous'): continue
            if relative.parts[0]=='continuous':
                assert record['name'].startswith('nested-refinement-')
                assert relative.parts[2] in ('refinement','refinement-24576')
            target=scratch/relative
            if target.exists():
                assert digest(target)==files[info.filename], f'Different existing packet at {target}'
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(payload)
    if exclusion: archive=archive/'ineligible'
    archive.mkdir(parents=True,exist_ok=True)
    name=record['name']+'.zip'
    assert Path(name).name==name
    target=archive/name
    if target.exists(): assert digest(target)==expected
    else: shutil.copyfile(packet,target)
    receipt={**record,'archive':name,'provenance':provenance,'files':files,
             'validation_eligible':exclusion is None,'exclusion_reason':exclusion}
    (archive/(record['name']+'.json')).write_text(json.dumps(receipt,indent=2))
    print(record['name'],len(files),'files verified; eligible',exclusion is None,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('packet',type=Path)
    p.add_argument('metadata',type=Path)
    p.add_argument('--scratch',type=Path,default=Path('results/nested-adaptive'))
    p.add_argument('--archive',type=Path,default=Path('archive/2026-11-25-nested-growth/validation'))
    a=p.parse_args()
    install(a.packet,a.metadata,a.scratch,a.archive)
