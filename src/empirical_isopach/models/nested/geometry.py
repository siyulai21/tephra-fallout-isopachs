"""Rank locations by the area of their score sublevel set on the authalic disk."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.probabilistic_isopach.global_volume import EarthDomain
from ..mass_mixture.geometry import RADIUS

DIAMETER = 2 * RADIUS
ANGLES = 48


@dataclass(frozen=True)
class Shape:
    family: str
    floor: float
    power: float
    second_angle: float = 0
    satellite_km: float = 0
    secondary_dominant: bool = False
    turn_deg: float = 0
    turn_radius_km: float = 1


def shapes():
    entries = [Shape("round", 1, 1), Shape("broad", .35, 1), Shape("narrow", .09, 3)]
    entries += [Shape("turning", .12, 2, turn_deg=turn, turn_radius_km=r)
                for turn in (-90,90) for r in (80,600)]
    entries += [Shape("two fans", .15, 2, a) for a in (75, -75, 145, -145)]
    entries += [Shape("detached", .18, 2, a, d, a == 135)
                for a in (-90, 45, 135) for d in (50, 250, 1250)]
    return tuple(entries)


SHAPES = shapes()


def reach(shape, theta):
    def lobe(angle):
        return np.maximum(0., .5 + .5*np.cos(angle))**shape.power
    main = lobe(theta)
    if shape.family == "two fans":
        main = np.maximum(main, .8*lobe(theta-np.deg2rad(shape.second_angle)))
    return shape.floor + (1-shape.floor)*main


def area_radius(shape, level, bearings=512):
    """Equivalent circular radius of a fan union a delayed secondary disk."""
    theta = (np.arange(bearings)+.5)*(2*np.pi/bearings)
    level = np.asarray(level)[:, None]
    main_delay = .12*shape.satellite_km if shape.secondary_dominant else 0.
    main = np.minimum(DIAMETER, np.maximum(0.,level-main_delay)*reach(shape, theta))
    squared_radius = np.mean(main**2,axis=1)
    if shape.satellite_km:
        d = shape.satellite_km
        delay = 0. if shape.secondary_dominant else .12*d
        radius = .42*np.maximum(level-delay, 0.)
        # Integrate only the angular arc subtended by the satellite. A small
        # distant summit can lie between all rays of a uniform compass grid.
        half_arc = np.where(radius<d,np.arcsin(np.minimum(1.,radius/d)),np.pi)
        angle = half_arc*(2*(np.arange(bearings)+.5)/bearings-1)
        main = np.minimum(DIAMETER,np.maximum(0.,level-main_delay)*
                          reach(shape,angle+np.deg2rad(shape.second_angle)))
        half = np.sqrt(np.maximum(radius**2-(d*np.sin(angle))**2, 0.))
        lower = np.maximum(main, np.clip(d*np.cos(angle)-half, 0., DIAMETER))
        upper = np.clip(d*np.cos(angle)+half, 0., DIAMETER)
        extra = np.maximum(upper-lower,0.)*(upper+lower)
        squared_radius += half_arc[:,0]/np.pi*np.mean(extra,axis=1)
    return np.sqrt(squared_radius)


@lru_cache(maxsize=None)
def rank_table(shape):
    levels = np.r_[0., np.geomspace(.005, DIAMETER/shape.floor*1.001, 768)]
    return levels, area_radius(shape, levels)


def positions(xy, latitude):
    axis = EarthDomain.wgs84(latitude).axis_scale
    p = np.asarray(xy)*[1/axis, axis]
    radius = np.linalg.norm(p, axis=1)
    if not np.isfinite(p).all() or np.any(radius > DIAMETER*(1+1e-10)):
        raise ValueError("Points outside the physical authalic disk")
    return radius, np.arctan2(p[:, 0], p[:, 1])


def ranks(radius, bearing, geometry):
    """Return sqrt(area/pi) in km for each geometry and location."""
    geom = np.asarray(geometry, dtype=int)
    out = np.empty((len(geom), len(radius)))
    for shape_id in np.unique(geom//ANGLES):
        shape = SHAPES[shape_id]
        mask = geom//ANGLES == shape_id
        theta = bearing - (geom[mask, None] % ANGLES)*(2*np.pi/ANGLES)
        # A rotation that depends on radius preserves r dr dtheta exactly.
        # The unrotated area table remains valid even for a curved fan.
        theta = theta-np.deg2rad(shape.turn_deg)*radius/(radius+shape.turn_radius_km)
        score = radius/reach(shape, theta)
        if shape.satellite_km:
            d = shape.satellite_km
            distance = np.sqrt(np.maximum(0., radius**2+d*d-2*d*radius*np.cos(
                theta-np.deg2rad(shape.second_angle))))
            main_delay, satellite_delay = ( .12*d, 0.) if shape.secondary_dominant else (0., .12*d)
            score = np.minimum(score+main_delay, satellite_delay+distance/.42)
        levels, table = rank_table(shape)
        out[mask] = np.interp(score, levels, table)
    return out


def geometry_probabilities():
    families = np.array([s.family for s in SHAPES])
    families_present, counts = np.unique(families, return_counts=True)
    per_shape = {k: 1/(len(counts)*n) for k,n in zip(families_present,counts,strict=True)}
    return np.repeat([per_shape[name]/ANGLES for name in families], ANGLES)
