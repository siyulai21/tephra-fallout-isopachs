"""Fixed 12,288-particle ceiling for the matched auxiliary validation matrix."""
from pathlib import Path

from src.empirical_isopach.models.nested.footprints.continuous import write_fit


def fit_part(folder,key,train,test,meta):
    folder=Path(folder)
    result=write_fit(folder,key,train,test,meta)
    if result['numerics']['passed']:
        return result
    return write_fit(folder/'refinement',key,train,test,meta,initial=12288,ceiling=12288)
