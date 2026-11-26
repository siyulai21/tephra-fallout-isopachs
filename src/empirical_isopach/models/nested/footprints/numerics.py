"""Fused double-precision likelihood, using SciPy's tail-stable normal CDF.

This evaluates the same robust likelihood without constructing multiple full
particle-by-site arrays. Endpoint differences are formed before subtracting the
predicted mean, preserving narrow interval widths in extreme proposals.
"""
import ctypes
import math

import numpy as np
from numba import njit
from numba.extending import get_cython_function_address
from ..geometry import ANGLES


_cdf = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_int)(
    get_cython_function_address('scipy.special.cython_special','__pyx_fuse_1log_ndtr'))
LOG_TWO_PI = math.log(2*math.pi)


@njit
def interval_event(lo, width):
    mid = lo+width/2
    if width*(1+abs(mid))<1e-4:
        return -.5*(mid*mid+LOG_TWO_PI)+math.log(width)+math.log1p((mid*mid-1)*width*width/24)
    if lo>0:
        left,right=-lo-width,-lo
    else:
        left,right=lo,lo+width
    high=_cdf(right,1)
    delta=_cdf(left,1)-high
    if delta<0:
        return high+math.log(-math.expm1(delta))
    # The standardized endpoints coincide only for unrepresentably small
    # widths relative to the location. Such negligible-likelihood proposals
    # still get a finite local limit rather than a spurious NaN.
    return -.5*(mid*mid+LOG_TWO_PI)+math.log(width)


@njit
def event(y,mu,sd,censored,lower,upper):
    z=(y-mu)/sd
    if math.isfinite(upper):
        return interval_event(z,(upper-y)/sd)
    if censored:
        return _cdf(-z if lower else z,1)
    return -.5*(z*z+LOG_TWO_PI)-math.log(sd)


@njit
def log_events(radius,pars,tail,y,known_sd,censored,lower,upper):
    out=np.zeros(len(radius))
    for i in range(len(radius)):
        h,length,drop,scale,p,noise=pars[:,i]
        for j in range(len(y)):
            x=radius[i,j]/length
            w=max(x-1.,0.)/scale
            loss=p*math.log1p(w/(p-1)) if tail[i] else w
            mu=h-drop*min(x,1.)-loss
            a=math.log(.95)+event(y[j],mu,math.hypot(noise,known_sd[j]),censored[j],lower[j],upper[j])
            b=math.log(.05)+event(y[j],mu,math.hypot(3*noise,known_sd[j]),censored[j],lower[j],upper[j])
            hi,lo=max(a,b),min(a,b)
            out[i] += hi+math.log1p(math.exp(lo-hi)) if math.isfinite(hi) else hi
    return out


@njit(cache=True)
def angular_interpolation(table,radius,bearing,geometry,core,jitter,turn):
    n,sites=len(geometry),len(radius)
    angles=table.shape[2]
    result=np.empty((n,sites))
    for i in range(n):
        shape=geometry[i]//ANGLES
        for j in range(sites):
            theta=bearing[j]-(geometry[i]%ANGLES)*2*math.pi/ANGLES-jitter[i]-turn[i]*radius[j]/(radius[j]+4*core[shape])
            angular=(theta*angles/(2*math.pi))%angles
            index=int(math.floor(angular))
            w=angular-index
            result[i,j]=table[shape,j,index]+w*(table[shape,j,(index+1)%angles]-table[shape,j,index])
    return result
