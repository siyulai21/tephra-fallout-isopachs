"""Independent geometric/analytic controls for complete-Earth integration."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pyproj import Geod
from scipy.special import roots_legendre

from src.Laea import make_laea_transform
from src.probabilistic_isopach.data import Observations
from src.probabilistic_isopach.distributions import LogMixture
from src.probabilistic_isopach.global_volume import EarthDomain, earth_cells, global_volume, infinite_plane_background


class RadialSurface:
    def __init__(self, latitude=36., tail='constant', length=800.):
        self.domain=EarthDomain.wgs84(latitude)
        self.tail=tail;self.length=length
        xy=np.array([[-200,-100],[-100,300],[100,-200],[400,100]])
        self.data=Observations(xy,np.zeros(4),np.zeros(4,bool),np.zeros(4),np.arange(4))

    def predict_log(self, xy):
        rho=self.domain.rho(xy)
        trend=(np.zeros(len(xy)) if self.tail=='constant' else
               -rho/self.length if self.tail=='exponential' else -2*np.log1p(rho/self.length))
        return np.log(2)+trend,np.full(len(xy),.4)


@pytest.mark.parametrize('latitude',[-90.,-73.,0.,36.,90.])
def test_full_earth_area_and_uniform_volume(latitude):
    model=RadialSurface(latitude)
    r=global_volume(model,latitude)
    earth=Geod(ellps='WGS84');e=np.sqrt(1-(earth.b/earth.a)**2)
    # Independent oblate-spheroid surface-area formula.
    area=2*np.pi*(earth.a/1000)**2*(1+(1-e*e)*np.arctanh(e)/e)
    assert r['converged']
    assert r['domain']['surface_area_km2']==pytest.approx(area,rel=1e-14)
    assert r['mean_km3']==pytest.approx(area*2*np.exp(.2)*1e-5,rel=1e-10)
    assert r['median_surface_km3']==pytest.approx(area*2*1e-5,rel=1e-10)
    assert r['fraction_on_opposite_hemisphere']==pytest.approx(.5,rel=1e-10)
    assert r['conditional_quantiles_km3'] is None


@pytest.mark.parametrize('tail',['exponential','power'])
def test_radial_integral_matches_closed_form_including_critical_power(tail):
    model=RadialSurface(tail=tail)
    r=global_volume(model,36.,tolerance=1e-8)
    R=model.domain.rho_max;L=model.length
    integral=L*L*(1-(1+R/L)*np.exp(-R/L)) if tail=='exponential' else L*L*(np.log1p(R/L)+1/(1+R/L)-1)
    assert r['converged']
    assert r['mean_km3']==pytest.approx(2*np.pi*2*np.exp(.2)*integral*1e-5,rel=1e-9)


@pytest.mark.parametrize('latitude',[-90.,-36.,0.,60.,90.])
def test_laea_ellipse_agrees_with_proj_and_area_jacobian(latitude):
    domain=EarthDomain.wgs84(latitude)
    fwd,inv,_=make_laea_transform(25.,latitude)
    edges=[0.,domain.rho_max/10,domain.rho_max]
    for xy,area in earth_cells(domain,edges,12,48,500):
        lon,lat=inv.transform(*xy.T)
        assert np.isfinite(lon).all() and np.isfinite(lat).all()
        np.testing.assert_allclose(np.column_stack(fwd.transform(lon,lat)),xy,atol=.0002,rtol=1e-7)
    # Independent geodetic integration: dA=a²(1-e²)/(1-e² sin²(phi))² d(sin(phi)) d(lambda).
    u,w=roots_legendre(128);earth=Geod(ellps='WGS84');e2=earth.f*(2-earth.f)
    lat=np.rad2deg(np.arcsin(u));lon=np.arange(256)*360/256-180
    LON,LAT=np.meshgrid(lon,lat);x,y=fwd.transform(LON,LAT)
    # Integral of squared normalized LAEA radius on a uniform disk is A/2.
    rho2=(x/domain.axis_scale)**2+(y*domain.axis_scale)**2
    area_weights=w*(earth.a/1000)**2*(1-e2)/(1-e2*u*u)**2*2*np.pi/256
    value=np.sum(rho2/domain.rho_max**2*area_weights[:,None])
    assert value==pytest.approx(domain.describe()['surface_area_km2']/2,rel=2e-9)


def test_mixture_integrates_expected_thickness_not_log_moments():
    a,b=RadialSurface(),RadialSurface()
    class Mixture:
        data=a.data
        def distribution(self, xy, *args):
            return LogMixture([.25,.75],np.array([np.full(len(xy),0.),np.full(len(xy),np.log(10.))]),np.zeros((2,len(xy))))
    r=global_volume(Mixture(),36.)
    area=r['domain']['surface_area_km2']
    assert r['mean_km3']==pytest.approx(area*(.25+.75*10)*1e-5,rel=1e-10)
    assert r['mean_km3']!=pytest.approx(area*np.exp(.75*np.log(10))*1e-5)


def test_unresolved_quadrature_is_reported_and_settings_are_validated():
    r=global_volume(RadialSurface(tail='power'),36.,tolerance=1e-15,orders=((2,4),(3,8),(4,16)))
    assert not r['converged']
    with pytest.raises(ValueError):EarthDomain.wgs84(91.)
    with pytest.raises(ValueError):global_volume(RadialSurface(),36.,tolerance=0)
    with pytest.raises(ValueError):list(earth_cells(EarthDomain.wgs84(0),[0,100],10,16,10))


def test_infinite_plane_background_has_analytic_directional_integral():
    model=SimpleNamespace(beta=np.array([np.log(2),3.,0.,0.]),amplitude=.4,radius_scale=100.,candidate=SimpleNamespace(tail='power'))
    r=infinite_plane_background(model)
    assert r['background_mean_km3']==pytest.approx(2*np.exp(.08)*100**2*2*np.pi/(2*1)*1e-5)
    model.beta[1]=2.
    assert infinite_plane_background(model)['background_mean_km3'] is None
    model.candidate.tail='exponential';model.beta[1]=3
    assert infinite_plane_background(model)['background_mean_km3']==pytest.approx(2*np.exp(.08)*100**2*2*np.pi/9*1e-5)


def test_global_result_is_independent_of_cli_diagnostic_bounds(tmp_path):
    from src.probabilistic_isopach.cli import main
    rows=[]
    for i,bounds in enumerate([['-400','-850','1200','1100'],['-1000','-1500','2000','2000']]):
        out=tmp_path/str(i)
        main(['Minoan','--output',str(out),'--no-publication','--grid','8','--volume-grid','8','--draws','20','--bounds',*bounds])
        rows.append(json.loads((out/'report.json').read_text()))
    assert rows[0]['model']==rows[1]['model']
    assert rows[0]['global_fallout']==rows[1]['global_fallout']
    assert rows[0]['volume']['mean_km3']!=rows[1]['volume']['mean_km3']
    assert rows[0]['primary_volume']=='global_fallout.selected'
