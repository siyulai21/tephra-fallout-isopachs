"""Every unsuccessful production partition receives the same refinement policy."""
import pytest

from src.empirical_isopach.models.nested.footprints import release


@pytest.mark.parametrize('passes,expected',[(768,[768]),(12288,[768,12288]),(24576,[768,12288,24576]),(None,[768,12288,24576])])
def test_uniform_refinement_stops_only_at_success_or_ceiling(tmp_path,monkeypatch,passes,expected):
    visited=[]
    def fit(folder,key,train,test,meta,*,initial=768,ceiling=6144):
        visited.append(initial)
        return {'numerics':{'passed':initial==passes}}
    monkeypatch.setattr(release,'write_fit',fit)
    moved=[]
    def rejuvenate(*args):
        moved.append(True)
        return {'numerics':{'passed':False}}
    monkeypatch.setattr(release.rejuvenation,'write_fit',rejuvenate)
    result=release.fit_part(tmp_path,'fold-0',None,None,{})
    assert visited==expected
    assert result['numerics']['passed']==(passes is not None)
    assert moved==([True] if passes is None else [])
