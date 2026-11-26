"""Resolve the displayed field and both conditional quartile surfaces.

The median volume can stabilize before angular features or uncertainty bands.
This check evaluates the same posterior quantiles at fixed independent spatial
nodes; it never filters or modifies an isopach.
"""
import numpy as np

from ..reach_integrals import nodes


def quantile_fields(model, xy):
    values = []
    for start in range(0, len(xy), 128):
        values.append(np.quantile(model.field(xy[start:start+128]), [.25,.5,.75], axis=0)/np.log(10.))
    return np.concatenate(values, axis=1)


def map_draws(model):
    grids = [nodes(12, seed, model.latitude)[:2] for seed in (18971,42533)]
    history, previous = [], None
    for requested in (1024,2048,4096,8192,16384):
        n = min(requested,len(model.z))
        fields = [quantile_fields(model.reduced(n),xy) for xy,_ in grids]
        volumes = np.mean([1.e-5*np.mean(weight*10.**field,axis=1)
                           for (_,weight),field in zip(grids,fields,strict=True)],axis=0)
        change, shape = None, None
        if previous is not None:
            change = np.abs(volumes/np.asarray(history[-1]['quartile_map_km3'])-1.)
            before, after = np.concatenate(previous,axis=1), np.concatenate(fields,axis=1)
            shape = []
            for k in range(3):
                visible = np.maximum(before[k],after[k]) >= np.log10(.25)
                shape.append(float(np.quantile(abs(after[k,visible]-before[k,visible]),.95)) if visible.any() else 0.)
        passed = bool(n>=2048 and change is not None and np.max(change)<.015 and max(shape)<.035)
        history.append(dict(draws=n,median_map_km3=float(volumes[1]),quartile_map_km3=volumes.tolist(),
                            relative_change=None if change is None else float(change[1]),
                            quartile_relative_change=None if change is None else change.tolist(),
                            p95_abs_log10_change_visible=None if shape is None else shape[1],
                            quartile_p95_abs_log10_change_visible=shape,passed=passed))
        if passed or n==len(model.z):
            break
        previous=fields
    return history
