"""Nested contours from positive, scale-dependent angular area growth.

The angular distribution describes an increment of deposit area, not a station
kernel or a rotated finished contour. Integrating positive increments preserves
nesting even while the dispersal axis turns. No station locations define shapes.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..geometry import ANGLES, DIAMETER
from . import geometry as legacy


@dataclass(frozen=True)
class Growth:
    family: str
    aspect: float = 1.
    focus: float = 0.
    core_km: float = 32.
    turn: float = 0.
    second: float = 0.
    distal_ratio: float = 0.


CORES = (2., 8., 32., 128., 512.)
SHAPES = (
    Growth("round"),
    *(Growth("single", a, f, c) for a in (1., 1.8, 3.)
      for f in (.2, .55, .85, .97) for c in CORES),
    *(Growth("turning", a, focus, c, turn=t) for a in (1.4, 2.4) for focus in (.55,.92)
      for t in (-100., -50., 50., 100.) for c in CORES),
    *(Growth("two phases", 1.8, focus, c, second=t, distal_ratio=f) for focus in (.65,.92)
      for t in (-145., -75., 75., 145.) for c in CORES for f in (0., .3, .8)),
    # Retain the detached alternative, including peaks away from the source.
    *(s for s in legacy.SHAPES if s.family == "detached"),
)
FAMILY_PRIOR = legacy.FAMILY_PRIOR
N_ANGLE = 384
N_LEVEL = 1025


def prior():
    names = np.array([s.family for s in SHAPES])
    return np.repeat([FAMILY_PRIOR[s] / (np.sum(names == s) * ANGLES) for s in names], ANGLES)


def angular_area(theta, aspect, focus):
    """Squared ray radius of a unit-area ellipse, with independent displacement.

    Mean angular area is one analytically: the ellipse has semiaxes a, 1/a
    and its centre is (a*focus, 0), with the source strictly inside it.
    """
    u, v = np.cos(theta)/aspect, np.sin(theta)*aspect
    k = u*u+v*v
    along = focus*u
    inside = 1.-focus*focus
    root = np.sqrt(along*along+k*inside)
    r = np.where(along >= 0., (along+root)/k, inside/(root-along))
    return r*r


@lru_cache(maxsize=None)
def fronts(shape, angles=N_ANGLE, levels_count=N_LEVEL):
    """Accumulate strictly positive area increments on a log-scale grid.

    Midpoint integration is in *area*, so each unclipped level has exactly
    its prescribed total area. Refinement changes only its allocation.
    """
    level = np.r_[0., np.geomspace(.0001, 256.*DIAMETER, levels_count-1)]
    theta = np.arange(angles)*(2*np.pi/angles)
    if shape.family == "round":
        return level, np.broadcast_to(level[:,None], (len(level), angles))
    c = shape.core_km
    circular = c*c*level*level/(c*c+level*level)
    directed = level*level-circular
    middle = np.sqrt((level[1:]**2+level[:-1]**2)/2.)
    turn = np.deg2rad(shape.turn)*middle/(middle+4.*c)
    development = middle/(middle+c)
    aspect = np.sqrt(shape.aspect)+(shape.aspect-np.sqrt(shape.aspect))*development
    focus = shape.focus*(.35+.65*development)
    kernel = angular_area(theta[None,:]-turn[:,None], aspect[:,None], focus[:,None])
    # Numerical angular quadrature is normalized independently of observations.
    kernel /= kernel.mean(axis=1,keepdims=True)
    core = angular_area(theta, np.sqrt(shape.aspect), .35*shape.focus)
    core /= core.mean()
    area = circular[:,None]*core
    area[1:] += np.cumsum(np.diff(directed)[:,None]*kernel, axis=0)
    if shape.family == "two phases":
        g = np.sqrt(directed)
        f = shape.distal_ratio
        h = f*g + (.8-f)*4.*c*g/(4.*c+g)
        secondary = angular_area(theta-np.deg2rad(shape.second), shape.aspect, shape.focus)
        secondary /= secondary.mean()
        area += h[:,None]**2*secondary
    return level, np.sqrt(np.maximum(area,0.))


@lru_cache(maxsize=None)
def lookup(shape):
    if shape.family == "detached":
        return legacy.lookup(shape)
    levels, front = fronts(shape)
    q = np.sqrt(np.mean(np.minimum(front,DIAMETER)**2,axis=1))
    radius = np.r_[0., np.geomspace(.001, DIAMETER, N_LEVEL-1)]
    values = np.column_stack([np.interp(radius,front[:,i],q) for i in range(N_ANGLE)])
    return levels, q, radius, values


def ranks(radius, bearing, geometry):
    radius, bearing = np.asarray(radius), np.asarray(bearing)
    geometry = np.asarray(geometry,dtype=int)
    out = np.empty((len(geometry),len(radius)))
    for i in np.unique(geometry//ANGLES):
        mask = geometry//ANGLES == i
        shape = SHAPES[i]
        if shape.family == "detached":
            old = legacy.SHAPES.index(shape)*ANGLES+geometry[mask]%ANGLES
            out[mask] = legacy.ranks(radius,bearing,old)
            continue
        if shape.family == "round":
            out[mask] = radius
            continue
        _, _, grid, values = lookup(shape)
        angular = ((bearing[None,:]-(geometry[mask,None]%ANGLES)*(2*np.pi/ANGLES))*(N_ANGLE/(2*np.pi)))%N_ANGLE
        low = np.floor(angular).astype(int)%N_ANGLE
        fraction = angular-np.floor(angular)
        high = (low+1)%N_ANGLE
        idx = np.clip(np.searchsorted(grid,radius,side="right")-1,0,len(grid)-2)
        w = (radius-grid[idx])/(grid[idx+1]-grid[idx])
        left = values[idx,low]+w*(values[idx+1,low]-values[idx,low])
        right = values[idx,high]+w*(values[idx+1,high]-values[idx,high])
        out[mask] = left+fraction*(right-left)
    return out
