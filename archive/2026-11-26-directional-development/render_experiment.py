import argparse,json
from pathlib import Path
import numpy as np
from explore import setup
from src.empirical_isopach.core.catalog import datasets
from src.empirical_isopach.core.audit import audited_data
from src.empirical_isopach.models.directional.multiscale.posterior import DirectionalField
from benchmarks.publication import result_home
from benchmarks.directional_scale_maps import read_grid,plate,windows
from src.empirical_isopach.visualization.deposit_maps import frames
from src.empirical_isopach.visualization.contour_limits import local_half_width

a=argparse.ArgumentParser();a.add_argument('variant');a.add_argument('--slugs',nargs='*');a.add_argument('--context',action='store_true');opt=a.parse_args()
setup(opt.variant)
root=Path('results/directional-geography-v6')
for r in datasets():
    if opt.slugs and r['slug'] not in opt.slugs:continue
    home=root/'experiments'/opt.variant/r['slug']/'full'
    if not (home/'field.npz').exists():continue
    grid=home/'grid.npz'
    if not grid.exists():
        model=DirectionalField.load(home/'field.npz')
        radii=np.r_[0,np.geomspace(.03,12750,337)]
        angles=np.arange(288)*2*np.pi/288
        xy=np.stack((model.domain.axis_scale*np.outer(radii,np.sin(angles)),np.outer(radii,np.cos(angles))/model.domain.axis_scale),axis=-1).reshape(-1,2)
        vals=model.central(xy).reshape(len(radii),len(angles))
        np.savez_compressed(grid,radii_km=radii,log10_cm=vals,axis=model.domain.axis_scale)
    old=result_home(r)/'directional';d,m=audited_data(r['path'])
    fs=[read_grid(old/'median-grid.npz'),read_grid(grid)]
    if opt.context: box=windows([json.loads((old/'deposit.json').read_text())])
    else:
        near=local_half_width(d)*np.array([-1,-1,1,1])
        box=windows([dict(full_bounds_km=frames(f,d,.5)[0],near_bounds_km=near) for f in fs])
    plate([(label,f,d,m,box) for label,f in zip(['Current principal',opt.variant],fs)],root/(r['slug']+'-'+opt.variant+'.png'),r['name']+' · matched observations and scales')
    print(r['slug'],flush=True)
