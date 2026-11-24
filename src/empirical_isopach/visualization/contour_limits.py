"""Data-enclosing isopachs and matched, contour-bearing near-vent windows."""
import contourpy
import numpy as np


def outer_contour(field, data):
    """Select the thickest doubling level enclosing at least 95% of accepted rows."""
    predicted = field.at_xy(*data.xy.T)
    if not len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("Every modeled observation must have a finite median on the map")
    required = (95*len(predicted)+99)//100
    cutoff = np.sort(predicted)[-required]
    exponent = int(np.floor(cutoff/np.log10(2.) + 1e-12))
    chosen = 2.**max(0, exponent)
    inside = predicted >= np.log10(chosen)-1e-12
    return dict(level_cm=chosen, target_fraction=.95, minimum_enclosed=required,
                accepted_rows=len(inside), inside=int(inside.sum()), target_met=bool(inside.sum()>=required),
                next_level_inside=int((predicted >= np.log10(2*chosen)-1e-12).sum()),
                row_ids=data.row_ids.tolist(), mapped_log10_cm=predicted.tolist())


def local_half_width(data):
    """Use the proximal cluster, with a 100 km ceiling when all sampling is distal."""
    distances = np.sort(np.linalg.norm(np.unique(data.xy, axis=0), axis=1))
    positive = distances[distances>1e-6]
    if not len(positive):
        return 1.
    reach = min(positive[:8][-1], 3*np.median(positive[:3]))
    width = np.clip(1.25*reach, 1., 100.)
    tick = 10.**np.floor(np.log10(width))
    return float(tick*np.ceil(width/tick))


def inner_contour(outer):
    predicted = np.array(outer["mapped_log10_cm"])
    exponent = np.floor(predicted.max()/np.log10(2.)+1e-12)
    level = float(2.**exponent)
    inside = predicted>=np.log10(level)-1e-12
    return dict(level_cm=level, inside=int(inside.sum()),
                row_ids=np.asarray(outer["row_ids"])[inside].tolist(),
                next_level_inside=int((predicted>=np.log10(level*2)-1e-12).sum()))


def near_levels(field,inner_cm):
    # The supported upper limit can fall below the normal half-centimetre colour scale.
    upper = int(np.rint(np.log2(inner_cm)))
    lower = -1 if upper>=0 else upper-1
    levels = 2.**np.arange(lower,upper+1)
    zmin,zmax = float(field.log10_cm.min()),float(field.log10_cm.max())
    return levels[(np.log10(levels)>zmin) & (np.log10(levels)<zmax)]


def visible_levels(field,half,levels):
    """Count actual drawable contour paths spanning at least an eighth of the panel width."""
    x = np.linspace(-half,half,301)
    z = field.on_grid(x,x)
    finite = z[np.isfinite(z)]
    if not len(finite):
        return []
    lines = contourpy.contour_generator(x=x,y=x,z=np.ma.masked_invalid(z))
    visible = []
    for level in levels:
        log_cm = np.log10(level)
        if finite.min()<log_cm<finite.max():
            pieces = lines.lines(log_cm)
            if any(np.linalg.norm(np.diff(piece,axis=0),axis=1).sum()>=.25*half for piece in pieces):
                visible.append(float(level))
    return visible


def near_frame(fields,data):
    """Choose the smallest tested square giving each fit two identifiable contours when available."""
    base = local_half_width(data)
    limits = [(field,near_levels(field,inner_contour(outer_contour(field,data))["level_cm"])) for field in fields]
    if not limits:
        raise ValueError("A near-vent frame requires a fitted field")
    cap = max(max(f.axis,1/f.axis)*f.radii[-1] for f in fields)
    # Nested 1–2–5 scale with intermediate sizes limits jumps between viewports.
    candidates = {base, max(base,cap)}
    for decade in range(int(np.floor(np.log10(base))),int(np.ceil(np.log10(cap)))+1):
        candidates.update(v for n in (1,1.25,1.5,2,2.5,3,4,5,6,8)
                          if base<(v:=n*10.**decade)<cap)
    best, best_score = base, (-1,-1)
    goals = [min(2,len(levels)) for _,levels in limits]
    for half in sorted(candidates):
        counts = [min(goal,len(visible_levels(f,half,levels))) if goal else 0
                  for (f,levels),goal in zip(limits,goals,strict=True)]
        score = (sum(count>=goal for count,goal in zip(counts,goals,strict=True)),sum(counts))
        if score>best_score:
            best,best_score = half,score
        if counts==goals:
            break
    return float(best)
