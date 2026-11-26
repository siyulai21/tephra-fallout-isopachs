"""A stable median must not hide unresolved uncertainty surfaces."""
import numpy as np

from src.empirical_isopach.models.nested.footprints import map_convergence


def test_quartile_changes_prevent_early_median_convergence(monkeypatch):
    monkeypatch.setattr(map_convergence,'nodes',lambda *args:(np.zeros((8,2)),np.ones(8)))

    class Field:
        latitude=0.
        def __init__(self,n=8192): self.z=np.empty((n,1))
        def reduced(self,n): return Field(n)
        def field(self,xy):
            width=.4 if len(self.z)==1024 else 1.
            return np.broadcast_to(np.linspace(-width,width,len(self.z))[:,None],(len(self.z),len(xy)))

    history=map_convergence.map_draws(Field())
    assert [h['draws'] for h in history]==[1024,2048,4096]
    assert history[1]['relative_change']<1e-12
    assert not history[1]['passed']
    assert history[2]['passed']
    assert max(history[1]['quartile_p95_abs_log10_change_visible'])>.035
