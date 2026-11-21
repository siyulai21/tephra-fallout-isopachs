"""Replay the two whole-deposit sensitivities against checksum-verified principal fields."""
import argparse
from pathlib import Path

from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.models.continuation_review import directional_review, spherical_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--directional-work",type=Path)
    args = parser.parse_args()
    path = resolve_dataset(args.case)
    spherical_review(path,args.results/"spherical-mixture",args.output/"spherical-mixture")
    work = args.directional_work or args.results/"directional/work/ensemble"
    directional_review(path,args.results/"directional",args.output/"directional",work)


if __name__ == "__main__":
    main()
