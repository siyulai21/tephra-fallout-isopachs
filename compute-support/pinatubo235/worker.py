"""Run the released Pinatubo methods on isolated hosted workers, reusing only verified checkpoints."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import gzip
import json
import shutil
import sys

from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.core.catalog import resolve_dataset

CONTROL = Path("compute-support/pinatubo235")
ROOT = Path("results/pinatubo-hosted")
SPHERE = ROOT / "spherical-mixture"
DIRECTIONAL = ROOT / "directional"
PATH = resolve_dataset("pinatubo-1991-layer-c")
NAME = "Pinatubo 1991 · climactic Layer C"
SEEDS = json.loads(gzip.decompress((CONTROL / "seed-index.json.gz").read_bytes()))
DATA, META = audited_data(PATH)
assert META["input_sha256"] == SEEDS["input_sha256"]
assert len(DATA.xy) == 235 and DATA.interval.sum() == 18


def checksum(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(512*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hydrate(folder, *, alternative=None):
    key = folder.relative_to(ROOT).as_posix() + "/"
    if not any(k.startswith(key) for k in SEEDS["raw"]) and alternative is not None:
        key = alternative.relative_to(ROOT).as_posix() + "/"
    if folder.exists() or not any(k.startswith(key) for k in SEEDS["raw"]):
        return
    folder.mkdir(parents=True)
    for filepath, raw in SEEDS["metadata"].items():
        if filepath.startswith(key):
            (folder / filepath[len(key):]).write_text(raw)
    for filepath, (oid, digest) in SEEDS["raw"].items():
        if filepath.startswith(key):
            source = CONTROL / "blobs" / (oid + ".npz")
            if checksum(source) != digest:
                raise ValueError("Seed bytes differ: " + filepath)
            shutil.copyfile(source,folder / filepath[len(key):])
    print("Restored verified original",key,"to",folder,flush=True)


def native_fit(folder, fn):
    try:
        return fn()
    except ValueError as exc:
        if not str(exc).startswith(("Stale posterior cache:", "Changed scientific inputs in ")):
            raise
        previous = Path("results/incompatible-inputs") / folder.relative_to(ROOT)
        previous.parent.mkdir(parents=True,exist_ok=True)
        folder.rename(previous)
        print("Native signature differed; retained original and started a fresh design",folder,flush=True)
        return fn()


def sphere(key):
    from src.empirical_isopach.core.validation import spatial_folds,training_mask
    from src.empirical_isopach.models.mass_mixture.source_protocol import protocol
    from src.empirical_isopach.models.mass_mixture.study import Sampling,fit_checked
    from src.empirical_isopach.validation import automatic_buffer
    baseline,alternatives=protocol("displaced")
    settings=Sampling()
    data=DATA
    if key.startswith("fold-"):
        fold=int(key.rsplit("-",1)[1])
        data=data.canonical()
        mask=training_mask(data,spatial_folds(data.xy,4),fold,automatic_buffer(data)[0])[0]
        data=data.subset(mask)
        prior=baseline
        settings=replace(settings,seed=settings.seed+fold+1)
        folder=SPHERE/"posteriors"/"validation"/key
    else:
        prior=next(p for p in (baseline,*alternatives) if p.name==key)
        folder=SPHERE/"posteriors"/key
    hydrate(folder)
    _,fit=native_fit(folder,lambda:fit_checked(data,META["vent_lonlat"][1],prior,folder,settings,progress=lambda m:print(m,flush=True)))
    print("Completed",key,"passed",fit["diagnostics"]["passed"],flush=True)


def directional(kind, index, fold=None):
    from src.empirical_isopach.models.directional.ensemble.members import MEMBERS,CODES,splits,directory
    from src.empirical_isopach.models.directional.geometry import Geometry,POLICIES,BASE
    from src.empirical_isopach.models.directional.study import SEED,fit,write_json
    work=DIRECTIONAL/"work"
    data=DATA.canonical()
    lat=META["vent_lonlat"][1]
    if kind=="policy":
        p=POLICIES[index]
        folder=work/"fits"/p.name
        seed=SEED+index
        hydrate(folder)
    else:
        reference=index<0
        p=BASE if reference else MEMBERS[index]
        code=0 if reference else CODES[index]
        folder=(work/kind/f"fold-{fold}" if reference else directory(work/"ensemble",p,kind,fold))
        if kind=="full":
            seed=SEED+code
            hydrate(folder,alternative=work/"fits"/p.name)
        else:
            data=next(tr for i,tr,te in splits(data,lat,kind) if i==fold)
            seed=SEED+code*1000+(101 if kind=="geographic" else 201)+fold
            if data is None:
                write_json(folder/"status.json",{"status":"empty"})
                return
            hydrate(folder)
    try:
        geometry=Geometry(data,lat,p)
    except ValueError as exc:
        if kind in ("policy","full"):
            raise
        write_json(folder/"status.json",{"status":"insufficient support","reason":str(exc)})
        return
    _,info=native_fit(folder,lambda:fit(data,geometry,seed,folder))
    if kind!="policy":
        write_json(folder/"status.json",{"status":"estimated",**info})
    print("Completed",p.name,kind,fold,info["passed"],flush=True)


def finish(family):
    if family=="directional":
        from src.empirical_isopach.models.directional.reporting import directional_dataset
        from src.empirical_isopach.models.continuation_review import directional_review
        directional_dataset(PATH,DIRECTIONAL,NAME,resume=True)
        directional_review(PATH,DIRECTIONAL,DIRECTIONAL/"sensitivity-study",DIRECTIONAL/"work/ensemble")
    else:
        from src.empirical_isopach.global_reporting import global_dataset
        from src.empirical_isopach.models.continuation_review import spherical_review
        global_dataset(PATH,SPHERE,NAME,resume=True,progress=lambda m:print(m,flush=True))
        spherical_review(PATH,SPHERE,SPHERE/"sensitivity-study")
    p=ROOT/family
    for f in ("deposit.json","deposit.png","median-grid.npz","extent/report.json","extent/support.json.gz","extent/heldout.csv.gz","sensitivity-study/report.json"):
        assert (p/f).is_file(),p/f
    print("Principal map and sensitivity review complete", family,flush=True)


if __name__=="__main__":
    op,*args=sys.argv[1:]
    if op=="sphere":sphere(args[0])
    elif op=="directional":directional(args[0],int(args[1]),int(args[2]) if len(args)>2 else None)
    elif op=="finish":finish(args[0])
    else:raise ValueError(op)
