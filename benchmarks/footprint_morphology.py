"""Compare unsmoothed isopachs, station-mask interventions, and whole-area shape controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull

from .nested_footprints import cases
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.extent_review import ExtentFields, ExtentView
from src.empirical_isopach.models.nested.continuation import load_packet
from src.empirical_isopach.models.nested.geometry import positions
from src.empirical_isopach.models.nested.reach import ReachDraws, Corridors
from src.empirical_isopach.models.nested.reach_integrals import nodes
from src.empirical_isopach.models.nested.report import MapView
from src.empirical_isopach.visualization.deposit import read_json
from src.empirical_isopach.visualization.deposit_field import open_field, sample_field, PolarField
from src.probabilistic_isopach.global_volume import EarthDomain

METHODS = ("finite corridors","original whole area","dispersal footprints","corridors before finite spread","intrinsic field of corridor draws")
LEVELS = .5*2.**np.arange(12)


def outlines(field, cm):
    z = field.log10_cm
    above = z>=np.log10(cm)
    if not above[0].all() or above[-1].any():
        return None
    j = np.arange(z.shape[1])
    # Keep the first exit from the vent, even when a ray intersects another patch farther away.
    i = np.argmax(~above,axis=0)-1
    fraction = (np.log10(cm)-z[i,j])/(z[i+1,j]-z[i,j])
    radial = field.radii[i]+fraction*(field.radii[i+1]-field.radii[i])
    directions = np.column_stack((np.sin(field.angles),np.cos(field.angles)))
    hull = ConvexHull(directions*radial[:,None]).equations
    products = directions @ hull[:,:2].T
    radius_hull = np.min(np.divide(-hull[:,2],products,out=np.full_like(products,np.inf),where=products>0.),axis=1)
    deficit = np.clip(1.-radial/radius_hull,0.,1.)
    southwest = (np.rad2deg(field.angles)>=202.5) & (np.rad2deg(field.angles)<247.5)
    v = np.log(radial)
    harmonics = np.fft.rfft(v-v.mean())/len(v)
    power = 2*abs(harmonics)**2
    if len(v)%2 == 0:
        power[-1] /= 2
    counts = []
    for prominence in (.10,.20):
        peaks,_ = find_peaks(np.tile(v,3),prominence=np.log1p(prominence))
        counts.append(int(np.sum((peaks>=len(v)) & (peaks<2*len(v)))))
    return dict(high_harmonic_log_r_rms=float(np.sqrt(power[5:].sum())),prominent_lobes_10pct=counts[0],
                prominent_lobes_20pct=counts[1],worst_indentation=float(deficit.max()),
                worst_SW_indentation=float(deficit[southwest].max()),median_SW_radius_km=float(np.median(radial[southwest])),
                vent_connected_fraction=float(min(1., np.pi*np.mean(radial**2)/field.area(cm))))


def paths(slug, rec, scratch):
    if slug=="mazama" and "published_input" in rec and cases()["mazama-published"]["path"]!=rec["path"]:
        if (scratch/"mazama300-corridor").is_dir():
            return scratch/"mazama300-corridor", scratch/"mazama300-whole"
        from .footprint_study import ARCHIVE
        reference = ARCHIVE/"mazama-300-controls"
        return reference/"corridor",reference/"whole"
    base = Path("eruptions")/rec["slug"]/"results"
    return base/"nested-isopachs",base/"nested-whole-area"


def intervention(model, train, meta, calibration, slug):
    lat = meta["vent_lonlat"][1]
    domain = EarthDomain.wgs84(lat)
    sub = model.reduced(384)
    r,theta = positions(train.xy,lat)
    trials = []
    for degrees in (0.,-22.5,22.5,45.):
        phi = theta + np.deg2rad(degrees)
        xy = np.column_stack((domain.axis_scale*r*np.sin(phi),r*np.cos(phi)/domain.axis_scale))
        design = train if degrees==0. else replace(train,xy=xy)
        shifted = ReachDraws(sub.parent,sub.law,Corridors.from_training(design,lat))
        fields = ExtentFields("nested-isopachs",shifted,design,lat,calibration["states"])
        view = ExtentView(fields,domain,calibration["weights"])
        trials.append((degrees,view))
    out = []
    for seed in (71261,91967):
        xy,w,_,_ = nodes(14,seed,lat)
        w /= len(xy)
        base = trials[0][1].central(xy)
        for deg,view in trials[1:]:
            shifted = view.central(xy)
            for cm in (.5,1.,4.,16.,64.):
                left,right = base>=np.log10(cm),shifted>=np.log10(cm)
                union = w[left|right].sum()
                out.append(dict(case=slug,degrees=deg,scramble=seed,level_cm=cm,
                                changed_area_km2=w[left^right].sum(),union_area_km2=union,
                                fraction_of_union_changed=w[left^right].sum()/union if union else None))
    return out


def run(job):
    slug,rec,scratch,out = job
    dest = out/slug
    dest.mkdir(parents=True,exist_ok=True)
    data,meta = audited_data(rec["path"])
    corr,wh = paths(slug,rec,scratch)
    old = read_json(corr/"extent/report.json")
    control = ReachDraws.load(corr/"full.npz").reduced(512)
    views = [open_field(corr,"nested-isopachs",data,meta,review=corr/"extent/report.json")[0],
             MapView(load_packet(wh/"full.npz").reduced(512)),
             open_field(scratch/"release"/slug,"nested-isopachs",data,meta)[0],
             MapView(control),MapView(control.parent)]
    rows = []
    for i,(label,view) in enumerate(zip(METHODS,views,strict=True)):
        # All families use the same polar nodes. Interleaved coarsening measures numerical sensitivity.
        field = sample_field(view,data,meta["vent_lonlat"][1],193,360)
        np.savez_compressed(dest/f"surface-{i}.npz",radii=field.radii,z=field.log10_cm,axis=field.axis)
        for stride in (1,2):
            g = PolarField(field.radii[::stride],field.log10_cm[::stride,::stride],field.axis)
            for cm in LEVELS:
                details=outlines(g,cm)
                rows.append(dict(case=slug,input_sha256=meta["input_sha256"],method=label,
                                 bearings=len(g.angles),level_cm=cm,area_km2=g.area(cm),
                                 vent_inside=bool(g.log10_cm[0,0]>=np.log10(cm)),**(details or {})))
    pd.DataFrame(rows).to_csv(dest/"morphology.csv",index=False)
    pd.DataFrame(intervention(control,data,meta,old["calibration"],slug)).to_csv(dest/"station-rotation.csv",index=False)
    print(slug,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--case",action="append")
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run,[(s,r,a.scratch,a.output) for s,r in cases().items() if not a.case or s in a.case]))
