"""Boundary-independent fallout volume on the complete WGS84 surface.

The ellipsoidal LAEA image of Earth is an ellipse, not the infinite plane.
Its area element is rho d(rho) d(theta). Only numerical resolution is refined;
no thickness threshold, map boundary or user-selected rectangle truncates mass.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import roots_legendre
from pyproj import Geod

from .distributions import thickness_summaries


@dataclass(frozen=True)
class EarthDomain:
    authalic_radius_km: float
    axis_scale: float
    vent_latitude: float

    @classmethod
    def wgs84(cls, latitude):
        if not np.isfinite(latitude) or abs(latitude) > 90:
            raise ValueError('Source latitude must be finite and between -90 and 90.')
        earth = Geod(ellps='WGS84')
        a, e2 = earth.a/1000, earth.f*(2-earth.f)
        e = np.sqrt(e2)
        def q(s):
            return (1-e2)*(s/(1-e2*s*s)+np.arctanh(e*s)/e)
        qp = q(1.)
        radius = a*np.sqrt(qp/2)
        phi = np.deg2rad(latitude)
        if abs(latitude) == 90:
            scale = 1.
        else:
            beta = np.arcsin(q(np.sin(phi))/qp)
            scale = a*np.cos(phi)/(np.sqrt(1-e2*np.sin(phi)**2)*radius*np.cos(beta))
        return cls(float(radius), float(scale), float(latitude))

    @property
    def rho_max(self):
        return 2*self.authalic_radius_km

    def rho(self, xy):
        return np.linalg.norm(np.asarray(xy)*[1/self.axis_scale, self.axis_scale], axis=-1)

    def arc_km(self, rho):
        """Distance on the authalic sphere, not an ellipsoidal geodesic."""
        return 2*self.authalic_radius_km*np.arcsin(np.clip(np.asarray(rho)/self.rho_max,0,1))

    def describe(self):
        return {'ellipsoid':'WGS84', 'source_latitude_degrees':self.vent_latitude,
                'authalic_radius_km':self.authalic_radius_km, 'axis_scale':self.axis_scale,
                'ellipse_semiaxes_km':[self.rho_max*self.axis_scale,self.rho_max/self.axis_scale],
                'surface_area_km2':float(np.pi*self.rho_max**2),
                'extent_rule':'Complete physical Earth; no user-selected integration boundary.',
                'projection_method':'Ellipsoidal Lambert azimuthal equal area, EPSG method 9820.'}


def earth_cells(domain, edges, radial_order, angular_order, radial_scale, phase=.5):
    """Positive product quadrature, log radial transform, exact Earth boundary.

    Yield one annulus at a time. Edges subdivide quadrature, never omit volume.
    """
    if (radial_order < 2 or angular_order < 4 or radial_scale <= 0
            or not np.isfinite(radial_scale) or not 0 <= phase < 1):
        raise ValueError('Invalid global quadrature settings.')
    edges = np.asarray(edges,float)
    if (len(edges)<2 or edges[0]!=0 or edges[-1]!=domain.rho_max
            or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0)):
        raise ValueError('Annuli must partition the entire Earth exactly once.')
    u, w = roots_legendre(radial_order)
    theta = (np.arange(angular_order)+phase)*2*np.pi/angular_order
    direction = np.column_stack((domain.axis_scale*np.cos(theta),np.sin(theta)/domain.axis_scale))
    for low, high in zip(edges[:-1],edges[1:]):
        t0,t1 = np.log1p(np.array([low,high])/radial_scale)
        t = t0+(u+1)*(t1-t0)/2
        rho = radial_scale*np.expm1(t)
        weight = w*(t1-t0)/2 *rho*radial_scale*np.exp(t)*2*np.pi/angular_order
        yield (rho[:,None,None]*direction[None,:,:]).reshape(-1,2), np.repeat(weight,angular_order)


def _integral(model, domain, edges, nr, nt, scale, phase=.5):
    rows=[]
    for xy, area in earth_cells(domain,edges,nr,nt,scale,phase):
        total=np.zeros(2)
        for start in range(0,len(xy),4096):
            median, expected, _ = thickness_summaries(model,xy[start:start+4096])
            with np.errstate(over='raise',invalid='raise'):
                total += np.array([np.exp(expected)@area[start:start+4096],
                                   np.exp(median)@area[start:start+4096]])*1e-5
        rows.append(total)
    return np.asarray(rows)


def global_volume(model, latitude, tolerance=.001,
                  orders=((24,64),(48,128),(96,256),(192,512),(384,1024))):
    """Full-Earth conditional expectations with refinement and angular-phase check.

    Reports no global probability interval: finite-rectangle simulations cannot
    be relabelled as global draws. Median-surface integral is not median volume.
    """
    if not np.isfinite(tolerance) or tolerance<=0 or len(orders)<3:
        raise ValueError('Use a positive tolerance and at least three refinement levels.')
    domain = EarthDomain.wgs84(latitude)
    observed_rho=domain.rho(model.data.xy)
    if not len(observed_rho) or not np.isfinite(observed_rho).all() or observed_rho.max()>=domain.rho_max:
        raise ValueError('Observations must lie inside the physical WGS84 projection domain.')
    farthest=float(observed_rho.max())
    if farthest<=0: raise ValueError('Observations need spatial extent.')
    # Observed-distance quartiles resolve the spatial correction; the farthest
    # site and hemisphere split allow exact regional contribution accounting.
    edges=np.unique(np.r_[0.,np.quantile(observed_rho,[.25,.5,.75,1.]),
                           np.sqrt(2)*domain.authalic_radius_km,domain.rho_max])
    scale=max(float(np.median(observed_rho[observed_rho>0])),1e-6)
    history=[];stable=0;phase_error=None;converged=False
    for nr,nt in orders:
        values=_integral(model,domain,edges,nr,nt,scale)
        totals=values.sum(axis=0)
        if not np.isfinite(totals).all() or np.any(totals<=0):
            raise FloatingPointError('Global integral is not finite and positive.')
        change=None if not history else float(np.max(np.abs(totals/np.array([
            history[-1]['mean_km3'],history[-1]['median_surface_km3']])-1)))
        # Also check annular contributions; total cancellation must not hide a
        # changing spatial distribution. Errors are relative to total volume.
        annulus_change=None if not history else float(np.max(np.sum(np.abs(values-previous),axis=0)/totals))
        history.append({'radial_order_per_annulus':nr,'angular_order':nt,
                        'mean_km3':float(totals[0]),'median_surface_km3':float(totals[1]),
                        'relative_change':change,'annulus_change_over_total':annulus_change})
        stable=stable+1 if change is not None and max(change,annulus_change)<tolerance else 0
        previous=values
        if stable>=2:
            shifted=_integral(model,domain,edges,nr,nt,scale,phase=.125)
            phase_error=float(np.max(np.sum(abs(values-shifted),axis=0)/totals))
            if phase_error<tolerance:
                converged=True;break
    annuli=[{'inner_authalic_arc_km':float(domain.arc_km(lo)),
              'outer_authalic_arc_km':float(domain.arc_km(hi)),
              'mean_km3':float(value[0]),'median_surface_km3':float(value[1])}
             for lo,hi,value in zip(edges[:-1],edges[1:],values)]
    beyond=float(values[edges[:-1]>=farthest,0].sum()/totals[0])
    opposite=float(values[edges[:-1]>=np.sqrt(2)*domain.authalic_radius_km,0].sum()/totals[0])
    return {'mean_km3':float(totals[0]),'median_surface_km3':float(totals[1]),
            'converged':converged,'relative_tolerance':tolerance,'history':history,
            'angular_phase_check_relative_error':phase_error,'domain':domain.describe(),
            'farthest_observation_authalic_arc_km':float(domain.arc_km(farthest)),
            'fraction_beyond_farthest_observation':beyond,
            'fraction_on_opposite_hemisphere':opposite,'annuli':annuli,
            'conditional_quantiles_km3':None,
            'scope':'Model-conditional global bulk fallout on the complete WGS84 Earth. Not DRE or all eruption products. No global probability interval is estimated.',
            'limitations':['Removing the arbitrary rectangle does not identify unobserved deposit structure.',
                'The existing planar trend and covariance are extrapolated through the equal-area projection; their global validity is an assumption.',
                'Fitted-parameter, source-position, preservation and model-selection uncertainty are not integrated.',
                'Refinement and phase agreement are numerical diagnostics, not rigorous error bounds or geological validation.']}


def infinite_plane_background(model):
    """Analytic unconditioned trend/background integral: tail stress, not volume.

    The conditioned GP correction is not included. The infinite plane extends
    beyond Earth, so this is never substituted for the physical-domain integral.
    """
    if hasattr(model,'models'):
        components=[infinite_plane_background(m) for m in model.models]
        total=None if any(c['background_mean_km3'] is None for c in components) else float(sum(
            w*c['background_mean_km3'] for w,c in zip(model.weights,components)))
        return {'background_mean_km3':total,'components':components,
                'scope':'Weighted infinite-plane background only; not a physical fallout volume.'}
    a,b,cx,cy=model.beta; c=np.hypot(cx,cy)
    floor=2. if model.candidate.tail=='power' else 0.
    gap=float(b-c-floor)
    integral=None
    if gap>0:
        if model.candidate.tail=='power':
            # Factored discriminants remain stable near the integrability boundary.
            angular=2*np.pi*(1/np.sqrt((b-2-c)*(b-2+c))-1/np.sqrt((b-1-c)*(b-1+c)))
        else:
            angular=2*np.pi*b/((b-c)*(b+c))**1.5
        integral=float(np.exp(a+model.amplitude**2/2)*model.radius_scale**2*angular*1e-5)
    return {'background_mean_km3':integral,'minimum_decay_minus_integrability_threshold':gap,
            'scope':'Unconditioned trend plus marginal GP variance on an infinite plane; conditioned spatial correction omitted. Diagnostic only.'}
