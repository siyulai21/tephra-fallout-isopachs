"""Smooth, area-preserving angular flow for continuous dispersal directions.

A source-relative rotation depending on radius is a bijection with unit
Jacobian in (r**2/2, theta). Its inverse maps nested regions to nested regions,
preserves the physical disk, and leaves every draw's area integral unchanged.
"""
import numpy as np
from scipy.special import ndtr

from ..geometry import ANGLES, DIAMETER
from . import growth


def shift(radius, geometry, warp):
    core = np.array([s.core_km for s in growth.SHAPES])[geometry//ANGLES]
    orientation = (2*ndtr(warp[:,0])-1)*(np.pi/ANGLES)
    turn = np.deg2rad(120.)*np.tanh(warp[:,1]/2.)
    return orientation[:,None]+turn[:,None]*radius[None,:]/(radius[None,:]+4*core[:,None])


def ranks(radius, bearing, geometry, warp):
    """Ranks after the inverse angular flow, with no modification of thickness."""
    radius,bearing=np.asarray(radius),np.asarray(bearing)
    geometry=np.asarray(geometry,dtype=int)
    out=np.empty((len(geometry),len(radius)))
    moved=bearing[None,:]-shift(radius,geometry,warp)
    for i in np.unique(geometry//ANGLES):
        mask=geometry//ANGLES==i
        shape=growth.SHAPES[i]
        if shape.family=='round':
            out[mask]=radius
            continue
        levels,q,grid,values=growth.lookup(shape)
        theta=moved[mask]-(geometry[mask,None]%ANGLES)*2*np.pi/ANGLES
        angular=(theta*growth.N_ANGLE/(2*np.pi))%growth.N_ANGLE
        low=np.floor(angular).astype(int)%growth.N_ANGLE
        fraction=angular-np.floor(angular)
        high=(low+1)%growth.N_ANGLE
        idx=np.clip(np.searchsorted(grid,radius,side='right')-1,0,len(grid)-2)
        w=(radius-grid[idx])/(grid[idx+1]-grid[idx])
        left=values[idx,low]+w*(values[idx+1,low]-values[idx,low])
        right=values[idx,high]+w*(values[idx+1,high]-values[idx,high])
        out[mask]=left+fraction*(right-left)
        if shape.family=='detached':
            d=shape.satellite
            relative=theta-np.deg2rad(shape.second)
            distance=np.hypot(radius-d*np.cos(relative),d*np.sin(relative))
            delay=0. if shape.satellite_first else .12*d
            out[mask]=np.minimum(out[mask],np.interp(delay+distance/.42,levels,q))
    return out


class TrainingRanks:
    """Precompute only the fixed-radius interpolation, leaving direction continuous."""
    def __init__(self,radius,bearing):
        self.radius,self.bearing=np.asarray(radius),np.asarray(bearing)
        self.table=[]
        for shape in growth.SHAPES:
            _,_,grid,values=growth.lookup(shape)
            idx=np.clip(np.searchsorted(grid,radius,side='right')-1,0,len(grid)-2)
            w=(radius-grid[idx])/(grid[idx+1]-grid[idx])
            self.table.append(values[idx]+w[:,None]*(values[idx+1]-values[idx]))
        self.table=np.asarray(self.table)
        self.base=self.at(np.zeros((len(growth.SHAPES)*ANGLES,8)),np.arange(len(growth.SHAPES)*ANGLES))

    def subset(self, indices):
        child=object.__new__(TrainingRanks)
        child.radius,child.bearing=self.radius[indices],self.bearing[indices]
        child.table=self.table[:,indices]
        child.base=self.base[:,indices]
        return child

    def at(self,z,geometry):
        from .numerics import angular_interpolation
        core=np.array([s.core_km for s in growth.SHAPES])
        jitter=(2*ndtr(z[:,6])-1)*(np.pi/ANGLES)
        turn=np.deg2rad(120.)*np.tanh(z[:,7]/2.)
        out=angular_interpolation(self.table,self.radius,self.bearing,geometry,core,jitter,turn)
        # The detached patch is an explicit distance, including at its summit.
        for i in np.unique(geometry//ANGLES):
            shape=growth.SHAPES[i]
            if shape.family=='detached':
                mask=geometry//ANGLES==i
                theta=self.bearing[None,:]-(geometry[mask,None]%ANGLES)*2*np.pi/ANGLES-shift(self.radius,geometry[mask],z[mask,6:])
                relative=theta-np.deg2rad(shape.second)
                d=shape.satellite
                distance=np.hypot(self.radius-d*np.cos(relative),d*np.sin(relative))
                delay=0. if shape.satellite_first else .12*d
                levels,q,_,_=growth.lookup(shape)
                out[mask]=np.minimum(out[mask],np.interp(delay+distance/.42,levels,q))
        return out
