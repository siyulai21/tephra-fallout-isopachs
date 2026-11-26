"""Joint posterior for thinning and intrinsic geometry, with an analytic volume for every draw."""
from pathlib import Path
from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
from scipy.special import ndtr

from .. import inference
from ..continuation import Continued, join, outer_status
from ..geometry import positions, ANGLES
from ..profile import parameters
from ..study import digest, engine_digest, write_json
from . import geometry
from . import growth, flow, flow_inference as sampling

LEGACY_PROTOCOL = "nested-dispersal-footprints-v2"
PROTOCOL = "nested-continuous-area-growth-v4"


@dataclass
class FlowField(Continued):
    geography_version: int = 4
    warp: np.ndarray | None = None

    @property
    def geography(self):
        return growth

    def ranks_at(self, xy):
        return flow.ranks(*positions(xy, self.latitude), self.geometry, self.warp)

    def field(self, xy):
        return self.log_at(self.ranks_at(xy))

    def subset(self, idx):
        return FlowField(**vars(super().subset(idx)), geography_version=4, warp=self.warp[idx])

    def recontinue(self, data, tilt=1.):
        return continued(self.z, self.geometry, self.power_tail, self.latitude, self.warp,
                         data, donor=self.donor, transfer=self.transfer, tilt=tilt)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as packed:
            return cls(**{k: (float(packed[k]) if k == "latitude" else int(packed[k]) if k == "geography_version" else packed[k]) for k in packed})


def engine():
    root = Path(__file__).parent
    sources = (engine_digest(), digest(root.parent/"continuation.py"), digest(root/"geometry.py"))
    sources += tuple(digest(root/name) for name in ("growth.py","flow.py","flow_inference.py","numerics.py","continuous.py"))
    return sha256("".join(sources).encode()).hexdigest()


def continued(z, geometry_ids, tail, latitude, warp, train, **kwargs):
    # join() asks for whole-area ranks. Give it a unique index per complete
    # geographic draw, then restore the actual intrinsic geometry identities.
    from ..inference import Deposit
    proxy = Deposit(z,np.arange(len(z)),tail,latitude)
    def ranker(radius,bearing,ids):
        return flow.ranks(radius,bearing,geometry_ids[ids],warp[ids])
    joined = join(proxy,train,ranker=ranker,**kwargs)
    joined.geometry = geometry_ids
    return FlowField(**vars(joined),geography_version=4,warp=warp)


def predictions(model, test):
    records = []
    # Small batches keep a large posterior over a densely sampled campaign affordable.
    for start in range(0, len(test.xy), 24):
        part = test.subset((np.arange(len(test.xy)) >= start) & (np.arange(len(test.xy)) < start+24))
        field = model.field(part.xy)
        sd = parameters(model.z)[-1][:,None]
        cdf = sum(w*ndtr((part.log_value-field)/np.hypot(k*sd, part.log_sd))
                  for k,w in ((1.,.95),(3.,.05))).mean(axis=0)
        from scipy.special import logsumexp
        score = logsumexp(inference.log_events(field,sd[:,0],part),axis=0)-np.log(len(sd))
        records.extend(dict(row_id=int(i), log_score=float(s), cdf=float(p), exact=bool(e))
                       for i,s,p,e in zip(part.row_ids, score, cdf, ~part.censored, strict=True))
        if getattr(model, "geography_version", 2) == 4:
            quantiles = np.quantile(field,[.025,.5,.975],axis=0)
            for j, record in enumerate(records[-len(part.xy):]):
                record.update(median_log_cm=float(quantiles[1,j]),
                              latent_lower95_log_cm=float(quantiles[0,j]),
                              latent_upper95_log_cm=float(quantiles[2,j]),
                              evidence="interval" if part.interval[j] else "upper" if part.censored[j] else "exact")
    return records


def fit_field(train, latitude, *, initial=768, ceiling=6144, seed=213091, on_attempt=None):
    n = initial
    attempts = []
    while True:
        raw, diagnostics = sampling.fit(train, latitude, initial=n, ceiling=n, seed=seed,
                                       ranker=growth.ranks, geometry_prior=growth.prior())
        model = continued(raw.z[:,:6],raw.geometry,raw.power_tail,latitude,raw.z[:,6:],train,seed=97231)
        check = outer_status(model)
        # Volume alone can agree while populations occupy different geographic
        # modes. Check latent thickness at the actual training sites as well.
        medians = np.array([np.median(part.field(train.xy),axis=0)
                            for part in (model.subset(i) for i in np.array_split(np.arange(len(model.z)),4))])
        site_spread = np.ptp(medians,axis=0)
        spatial = dict(median_log_cm=float(np.median(site_spread)),
                       p95_log_cm=float(np.quantile(site_spread,.95)),
                       passed=bool(np.median(site_spread)<.15 and np.quantile(site_spread,.95)<.4))
        parent = diagnostics["attempts"][-1]
        # Evidence and every independent population are kept through the refinement ceiling.
        attempts.append(dict(particles_per_population=n, intrinsic_passed=parent["passed"],
                             intrinsic_volume_quantiles=parent["population_volume_quantiles_km3"],
                             log_evidence_exponential_power=parent["log_evidence_exponential_power"],
                             tempering=parent["tempering"],
                             spatial_populations=spatial,
                             continuation=check))
        if on_attempt is not None:
            on_attempt(model,attempts[-1])
        if (parent["passed"] and check["passed"] and spatial["passed"]) or n >= ceiling:
            break
        n = min(2*n, ceiling)
    return model, dict(seed=seed, initial=initial, ceiling=ceiling, attempts=attempts,
                       passed=bool(parent["passed"] and check["passed"] and spatial["passed"]))


def write_fit(folder, key, train, test, meta, *, initial=768, ceiling=6144, seed=213091):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info, packet = folder/(key+".json"), folder/(key+".npz")
    identity = dict(protocol=PROTOCOL, engine_sha256=engine(), input_sha256=meta["input_sha256"],
                    train=train.row_ids.tolist(), test=[] if test is None else test.row_ids.tolist(),
                    initial=initial, ceiling=ceiling, seed=seed)
    if info.exists():
        r = json.loads(info.read_text())
        if r["identity"] != identity or digest(packet) != r["posterior_sha256"]:
            raise ValueError(f"Posterior, input, or run specification differs at {info}")
        return r
    progress = folder/(key+"-progress.json")
    checkpoints = []
    if progress.exists():
        saved = json.loads(progress.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Checkpoint input or engine differs at {progress}")
        checkpoints = saved["checkpoints"]
        if any(digest(folder/c["packet"]) != c["posterior_sha256"] for c in checkpoints):
            raise ValueError(f"Checkpoint posterior changed at {progress}")

    def complete(attempt):
        return bool(attempt["intrinsic_passed"] and attempt["continuation"]["passed"]
                    and attempt["spatial_populations"]["passed"])

    def checkpoint(model,attempt):
        dest = folder/"attempts"/f"{key}-{attempt['particles_per_population']}.npz"
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,**vars(model))
        checkpoints.append(dict(packet=dest.relative_to(folder).as_posix(),posterior_sha256=digest(dest),numerics=attempt))
        write_json(progress,dict(identity=identity,checkpoints=checkpoints))
        print(folder.name,key,attempt["particles_per_population"],"particles per population; passed",complete(attempt),flush=True)

    old = [c["numerics"] for c in checkpoints]
    if old and (complete(old[-1]) or old[-1]["particles_per_population"] >= ceiling):
        model = FlowField.load(folder/checkpoints[-1]["packet"])
        checks = dict(seed=seed,initial=initial,ceiling=ceiling,attempts=old,passed=complete(old[-1]))
    else:
        start = min(2*old[-1]["particles_per_population"],ceiling) if old else initial
        model, checks = fit_field(train, meta["vent_lonlat"][1], initial=start, ceiling=ceiling,
                                  seed=seed,on_attempt=checkpoint)
        checks.update(initial=initial,attempts=old+checks["attempts"])
    np.savez_compressed(packet, **vars(model))
    draws = model.volume()
    names = [s.family for s in growth.SHAPES]
    modes = np.array(names)[model.geometry // ANGLES]
    r = dict(identity=identity, posterior_sha256=digest(packet), numerics=checks, draws=len(draws),checkpoints=checkpoints,
             volume_quantiles_km3=np.quantile(draws, [.025,.5,.975]).tolist(),
             posterior_family={f:float(np.mean(modes==f)) for f in geometry.FAMILY_PRIOR},
             heldout=predictions(model,test) if test is not None else [])
    write_json(info,r)
    return r
